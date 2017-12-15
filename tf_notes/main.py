# test for creating my own network
# using tensorflow

import tensorflow as tf
import numpy as np
import os
import sys

import tf_hlconvnet as hl

# constants for net trival settings


def main(args):
    net = hl.hlconvnet()
    net.offline_supervised('/home/hiroki/ships/raw_data/train/')
    net.offline_unsupervised('/home/hiroki/ships/raw_data/unlabeled/')
    net.train(1, 1, 1)

tf.app.run()

