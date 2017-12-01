# tf_hlconvnet.py
# Hybrid Learning Convolutional Network implementation
# Algorithm and implementation are intrinscally authorized
# by Chao Xiang

import numpy as np
import tensorflow as tf
import tf_hlconv as hl

class hlconvnet(object):
    def __init__(self):
        # a batch of images [nbatch, height, width, channel]
        self.graph = tf.Graph()
        # how to run unsupervised training and the supervised
        with self.graph.as_default():
            self.x = tf.random_uniform([16,5,5,3])
            feeds = tf.random_uniform([16,4,4,5])
            ksizes = [1,2,2,1]
            uw,sw = hl.hlconv_make_params(
                    n = 5,                        # number of filters 
                    depth = self.x.shape.as_list()[3],
                    ksizes = ksizes               # kernel sizes
            ) # create unsupervised and supervised training variables
            self.y,self.utrain_op = hl.hlconv(
                    x = self.x,
                    uw = uw,
                    sw = sw,
                    ksizes = ksizes,
                    strides = [1,1,1,1],
                    padding = 'VALID',
                    name = 'hlconv-1'
            )
            self.opt = tf.train.AdagradOptimizer(0.01)
            self.loss = tf.reduce_sum(tf.square(self.y - feeds))
            self.strain_op = self.opt.minimize(self.loss)
            self.sess = tf.Session()
            # initialization
            print(self.sess.run(tf.global_variables_initializer()))
            # the forward pass
            print(self.sess.run(self.y))
            # the unsupervised training
            print(self.sess.run(self.utrain_op))
            # run the supervised training
            print(self.sess.run(self.strain_op))


# test for class
def main(args):
    net = hlconvnet()

tf.app.run()
