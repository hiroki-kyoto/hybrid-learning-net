#!/bin/python
import numpy as np
import sys
import os

num = 30
bf = file('BPNN.LOG')
hf = file('HLNN.LOG')
bpnn = np.zeros(num)
hlnn = np.zeros(num)
for i in xrange(num):
    bpnn[i] = float(bf.readline())
    hlnn[i] = float(hf.readline())
# print result of each model
print "BPNN MODEL RESULT: ", bpnn
print "HLNN MODEL RESULT: ", hlnn
# calculate the mean score of each model
bpmean = bpnn.sum()/len(bpnn)
hlmean = hlnn.sum()/len(hlnn)
print "MEAN CORRECTNESS OF BPNN: ", bpmean
print "MEAN CORRECTNESS OF HLNN: ", hlmean
# calculate the square error of each model
print "SQUARE ERROR OF BPNN: ", ((bpnn-bpmean)*(bpnn-bpmean)).sum()/len(bpnn)
print "SQUARE ERROR OF HLNN: ", ((hlnn-hlmean)*(hlnn-hlmean)).sum()/len(hlnn)
