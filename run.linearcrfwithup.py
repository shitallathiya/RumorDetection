#!/usr/bin/python
# -*- coding: utf-8 -*-
# created by : Arkaitz Zubiaga
# modified by :shital lathiya

import numpy as np
import csv
from scipy.sparse import csc_matrix
from sklearn import metrics
import sys, os
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import nltk, json, re
from pprint import pprint
import htmlentitydefs
from os import listdir
from os.path import isfile, isdir, join, exists
import time
import gensim
import numpy as np
from collections import OrderedDict
import datetime

from pystruct.models import ChainCRF
import pystruct.learners as learners

import evaluation

def get_feature_ids(features1, features2):
    featids = {}
    featid = 0

    for tweetid, feats in features1.iteritems():
        for feat, value in feats.iteritems():
            if not feat in featids:
                featids[feat] = featid
                featid += 1

    for tweetid, feats in features2.iteritems():
        for feat, value in feats.iteritems():
            if not feat in featids:
                featids[feat] = featid
                featid += 1

    return featids

def get_features_optimised(datasets, targetdataset, features):
    vectors = []
    gts = []
    ids = []
    uids=[]

    for dataset in datasets:
        dvectors = []
        dgts = []
        dids = []
        duids=[]

        thlist = {}
        userlist = {}
        folders = [rtype for rtype in listdir(join(datasetspath, dataset)) if isdir(join(datasetspath, dataset, rtype))]
        for folder in folders:
            gt = rumourtypes[folder]
            tweetfiles = [tweetfile for tweetfile in listdir(join(datasetspath, dataset, folder))]
            for tweetfile in tweetfiles:
                with open(join(datasetspath, dataset, folder, tweetfile)) as fh:
                    tweet = json.load(fh)
                tweetid = tweet["id_str"]
                userid = tweet["user"]["id"]
                thlist[tweetid] = gt
                userlist[tweetid] = userid
        othlist = OrderedDict(sorted(thlist.items()))
        ouserlist=OrderedDict(sorted(userlist.items()))

        # for tweetid,userid in ouserlist.iteritems():
        #     duids.append(userid)

        for tweetid, gt in othlist.iteritems():
            vector = []

            dids.append(tweetid)
            dgts.append(gt)
            with open('features/' + targetdataset + '/' + tweetid, 'r') as fh:
                featvalues = fh.read().split(' ')
                for featvalue in featvalues:
                    if '::' in featvalue:
                        featvalue = featvalue.split('::')
                    else:
                        featvalue = featvalue.split(':')
                    if featvalue[0] in features:
                        try:
                            vector.append(float(featvalue[1]))
                        except:
                            print str(tweetid)
                            print(featvalue[0])
                            print(featvalue[1])
                            sys.exit(1)
                # print "vector before rumour ratio:",vector
                with open('features/' + targetdataset + '/thisevent_users_with_RR_status.csv', 'r') as userfile:
                    userid1=ouserlist[tweetid]
                    # print "tweetid:",tweetid,"userid:",userid1
                    filereader = csv.DictReader(userfile)
                    userRR_OR_followingRR=2
                    for row in filereader:
                          while (int(row['userid']) == userid1):
                              userRR_OR_followingRR=row['userRR_OR_avgfollowingRR']
                              vector.append(float(userRR_OR_followingRR))
                              # print "rumour ratio for existing userid ",userid,"tweetid ",tweetid,"in dataset ",dataset,"is : ", rumourratio
                              break;
                          else:
                            continue;
                          break;

                    # "value of rumour ratio is: ", rumourratio
                    if userRR_OR_followingRR==2:
                        # print "rumour ratio for userid", userid1, "not in users with tweetid ", tweetid, "in dataset ", dataset, "is : ", rumourratio
                        vector.append(0.0)


            dvectors.append(np.array(vector))
            # print "dvector:", dvectors

        vectors.append(np.array(dvectors))
        # print "vectors:", vectors

        gts.append(np.array(dgts))
        ids.append(np.array(dids))
        # uids.append(np.array(duids))
    return (np.array(vectors), np.array(gts), np.array(ids), ouserlist)


def set_category_ids(gts1, gts2):
    catid = 0
    catids = {}
    catnames = {}

    for tweetid, gt in gts1.iteritems():
        if not gt in catids:
            catid += 1
            catids[gt] = catid
            catnames[catid] = gt
            gts1[tweetid] = catid
        else:
            gts1[tweetid] = catids[gt]

    for tweetid, gt in gts2.iteritems():
        if not gt in catids:
            catid += 1
            catids[gt] = catid
            catnames[catid] = gt
            gts1[tweetid] = catid
        else:
            gts1[tweetid] = catids[gt]

    return (catnames, gts1, gts2)

def build_matrix(features):
    rows = []
    cols = []
    values = []
    rowcount = 0

    for tweetid, feats in features.iteritems():
        for feat, value in feats.iteritems():
            featid = featids[feat]

            rows.append(rowcount)
            cols.append(int(featid))
            values.append(value)

        rowcount += 1

    row = np.asarray(rows)
    col = np.asarray(cols)
    data = np.asarray(values)
    return csc_matrix((data, (row, col)), shape=(rowcount, featcount))

if len(sys.argv) < 3:
    print("")
    print("    No dataset and features specified, e.g.:")
    print("")
    print("    python run.classifier.from.twitter.json.py ferguson w2v")
    print("")
    sys.exit()

start_time = time.time()
print"start time:", start_time
print("running CRFUP with Content,Social features")

datasets = []
if "," in sys.argv[1]:
  data = sys.argv[1].split(",")
  for d in data:
    datasets.append(d)
else:
  datasets.append(sys.argv[1])

features = sorted(sys.argv[2].split(','))

outfile = 'linearcrf-' + '-'.join(features)
if exists('results/' + outfile):
  print('Done earlier!')
  sys.exit()

all_test_gt = []
all_predicted = []

rumourtypes = {'rumours': 1, 'non-rumours': 0}
id_preds = {}
id_gts = {}
for dataset in datasets:
  datasetspath = 'threads'

  if not os.path.exists(join(datasetspath, dataset)):
    print('Dataset not found.')
    sys.exit()

  traindatasets = [f for f in listdir(datasetspath) if f != dataset]

  datasetbase = dataset
  if '-' in dataset:
    dstokens = dataset.split('-')
    datasetbase = dstokens[0]

  train_features, train_gt, train_ids,train_uids = get_features_optimised(traindatasets, dataset, features)

  test_features, test_gt, test_ids,test_uids = get_features_optimised([dataset], dataset, features)

  train_data = train_features
  test_data = test_features

  classcounts = [0, 0]
  for thread in train_gt:
    for gt in thread:
      classcounts[gt] += 1

  classweights = [np.amin(classcounts) / float(x) for x in classcounts]
  model = ChainCRF(directed=True, inference_method="ad3", class_weight=classweights)

  ssvm = learners.FrankWolfeSSVM(model=model, max_iter=5000, C=1)
  ssvm.fit(train_data, train_gt)
  y_pred = ssvm.predict(test_data)

  acc = 0
  items = 0
  for k, thread_ids in enumerate(test_ids):
    thread_gts = test_gt[k]
    thread_preds = y_pred[k]

    for m, tweetid in enumerate(thread_ids):
      if not int(tweetid) in id_preds:
        if thread_preds[m] == 4:
          thread_preds[m] = 3
        id_preds[int(tweetid)] = thread_preds[m]
        id_gts[int(tweetid)] = thread_gts[m]

        items += 1
        if thread_preds[m] == thread_gts[m]:
          acc += 1

  acc = acc / float(items)
  print(outfile + ' -- ' + dataset + ': accuracy is ' + str('%.4f' % acc))

  for value in test_gt:
    all_test_gt.append(value)
  for value in y_pred:
    all_predicted.append(value)

with open('output.tsv', 'w+') as fw:
  for tweetid, pred in id_preds.iteritems():
    fw.write(str(tweetid) + '\t' + str(pred) + '\n')

evaluation.evaluate(id_preds, id_gts, 'linearcrf', features)

end_time = time.time()
print"end time:",end_time
temp = end_time-start_time
hours = temp//3600
temp = temp - 3600*hours
minutes = temp//60
seconds = temp - 60*minutes
print('Total Duration- %d:%d:%d' %(hours,minutes,seconds))