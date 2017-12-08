# tf_hlconvnet.py
# Hybrid Learning Convolutional Network implementation
# Algorithm and implementation are intrinscally authorized
# by Chao Xiang

import numpy as np
import tensorflow as tf
import tf_hlconv as hl

# configurations
N_DIGITS = 8
N, H, W, C = 1, 256, 256, 3

class hlconvnet(object):
    def __init__(self):
        # a batch of images [nbatch, height, width, channel]
        self.graph = tf.Graph()
        # how to run unsupervised training and the supervised
        with self.graph.as_default():
            # data feed in and back
            self.x = tf.placeholder(
                    dtype=tf.float32, 
                    shape=[N, H, W, C]
            )
            self.y = tf.placeholder(
                    dtype=tf.int32,
                    shape=[N, 1]
            )

            # containers for layers and parameters
            self.hlc = [] # hybrid conv operators
            self.uws = [] # unsupervised weights
            self.sws = [] # supervised weights
            self.sbs = [] # supervised biases
            self.uts = [] # unsupervised training ops
            self.layers = [] # layers 

##################### LAYER #1 ########################

            # generate the first hlconv layer
            # with kernel size 2x2,
            # with kernel number 256
            ksizes = [1,3,3,1]
            n = 256
            
            uw,sw = hl.hlconv_make_params(
                    n = n,
                    depth = self.x.shape.as_list()[3],
                    ksizes = ksizes
            )
            
            self.uws.append(uw)
            self.sws.append(sw)
            
            hlconv, utrain_op = hl.hlconv(
                    x = self.x,
                    uw = uw,
                    sw = sw,
                    ksizes = ksizes,
                    strides = [1,1,1,1],
                    padding = 'VALID',
                    name = 'hlconv-1'
            )
            
            self.hlc.append(hlconv)
            self.uts.append(utrain_op)

            # add bias
            bias = tf.Variable(
                    tf.random_uniform(
                        [hlconv.shape.as_list()[0]]
                    )
            )
            add_bias = tf.nn.bias_add(
                    hlconv,
                    bias
            )
            self.sbs.append(bias)

            # apply activation function
            apply_act = tf.nn.relu(add_bias)
            self.layers.append(apply_act)

#################### LAYER #2 ########################

            # the final layer is still a hlconv
            # however it is an inception operator
            # it must contains $N_DIGITS filters
            # to convert the tensor into output shape
            ksizes = [1,1,1,1]
            n = N_DIGITS
            uw, sw = hl.hlconv_make_params(
                    n = n,
                    depth = hlconv.shape.as_list()[3],
                    ksizes = ksizes
            )

            self.uws.append(uw)
            self.sws.append(sw)

            hlconv, utrain_op = hl.hlconv(
                    x = apply_act,
                    uw = uw,
                    sw = sw,
                    ksizes = ksizes,
                    strides = [1,1,1,1],
                    padding = 'VALID',
                    name = 'hlconv-2'
            )

            self.hlc.append(hlconv)
            self.uts.append(utrain_op)

            # add bias
            bias = tf.Variable(
                    tf.random_uniform(
                        [hlconv.shape.as_list()[0]]
                    )
            )
            self.sbs.append(bias)
            add_bias = tf.nn.bias_add(bias)
            
            # apply activation function
            apply_act = tf.nn.relu(add_bias)
            self.layers.append(apply_act)

####################### OUTPUT LAYER #######################
            # using maxpooling to get the output vector
            ksizes = [
                    1,
                    apply_act.shape.as_list()[1],
                    apply_act.shape.as_list()[2],
                    1
            ]
            self.logits = tf.nn.max_pool(
                    apply_act,
                    ksizes = ksizes,
                    strides = ksizes,
                    padding = 'VALID'
            )
            self.logits = tf.reshape(self.logits, [N_DIGITS])

################### PREDICTION AND TRAINING ##############

            # get predicted label and its probability
            self.pred = tf.argmax(logits, 1)
            self.prob = tf.nn.softmax(logits)

            # for training, convert from labels into one-hot vector
            self.onehot = tf.one_hot(
                    tf.cast(labels, tf.int32), 
                    N_DIGITS,
                    1,
                    0
            )

            # training loss
            #self.loss = tf.reduce_sum(tf.square(self.y - feeds))
            self.loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=self.onehot,
                    logits=self.logtis
            )
            
            # optimizer for supervised learning
            self.opt = tf.train.AdagradOptimizer(learning_rate=0.01)

            # the only supervised training operator
            self.st = self.opt.minimize(self.loss)

    # @nalt : number of alternation
    # @uepo : unsupervised iterations within an epoch
    # @sepo : supervised iterations within an epoch
    def train(self, nalt, uepo, sepo):
        # initialization
        self.sess.run(tf.global_variables_initializer())
        for i in xrange(nalt):
            for ui in xrange(uepo):
                for uti in xrange(len(self.uts)):
                    self.sess.run(
                            self.uts[uti], 
                            feed_dict={
                                self.x:,
                                self.y:
                                })
            for si in xrange(sepo):
                self.sess.run(
                        self.st,
                        feed_dict={
                            self.x:,
                            self.y:
                        })
            # for each alternation, print the training losses
            if __debug__:
                self.sess.run(self.loss) # need another op to calc
                for uti 
                self.sess.run(self.enery) # need a grou of ops

# test for class
def main(args):
    net = hlconvnet()

tf.app.run()
