# test for creating my own network
# using tensorflow

import tensorflow as tf
import numpy as np
import os
import sys

# constants for net trival settings
N_CLASS = 28    # number of classes
X_FEATURE = 'x' # name of input feature

def make_hybrid_learning_net(features, labels, mode):
    ''' Building a hybrid learning net'''
    x = features[X_FEATURE]
    input_shape = x.get_shape().as_list()
    
    # only image dataset allowed: 4D tensor
    assert len(input_shape)==4
    
    # build hybrid learning conv layer
    with tf.variable_scope('hl_conv_1'):
        hl_conv_1_embed = tf.constants

def main():
    # the feed-in layer
    # using SHIP dataset
    

