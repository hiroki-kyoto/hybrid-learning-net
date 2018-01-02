# -*- coding:utf8 -*-
# test for creating my own network
# using tensorflow

import tensorflow as tf
import numpy as np
import os
import sys

import tf_hlconvnet as hl

import setting as st


def main(argv):
    if len(argv)!=2:
        print 'HELP MESSAGE:'
        print '\t\t', 'python test.py [model_path]'
        exit()
    
    # restore label list from files
    label_file = open(st.__JSON_LABEL_LIST__,'rt')
    label_json = label_file.read()
    label_list = label_json.decode('utf8')
    
    net = hl.hlconvnet(N=1) # N is the batch
    conf_file = open(st.__JSON_CONF_FILE__, 'rt')
    conf_json = conf_file.read()
    net.build_graph(conf_json)
    net.restore(argv[1])
    
    # for each class, get its accuracy
    path = st.__DATA_PATH__%st.__TEST_DIR__
    subdirs = os.listdir(path)
    labels = []
    
    for _subdir in subdirs:
        labels.append(_subdir.decode('utf8'))
    
    labels.sort()
    stat_real = np.zeros([len(labels)])
    stat_pred = np.zeros([len(labels)])

    for _i in xrange(len(labels)):
        label = labels[_i]
        print label
        _subdir = os.path.join(path, label)
        pred, prob = net.predict(_subdir)
        correct = 0.0
        tot = len(pred)
        stat_real[_i] = tot

        for _ii in xrange(tot):
            stat_pred[pred[_ii][0]] += 1.0

            #pred_label = label_list[pred[_ii][0]]
            #pred_label = pred_label.decode('utf-8')
            #if pred_label==label:
            #    correct += 1.0

            if pred[_ii][0] == _i:
                correct += 1.0

        print correct/tot

    # show statistics 
    print stat_real
    print stat_pred

tf.app.run()

