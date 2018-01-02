# tf_hlconvnet.py
# Hybrid Learning Convolution Network implementation
# Algorithm and implementation are intrinsically authorized
# by Chao Xiang

# -*- coding:utf8 -*-

import numpy as np
import PIL.Image as pi

import tensorflow as tf
import tf_hlconv as hl

import os
import time

import json

class hlconvnet(object):
    def __init__(self, **kwargs):
        self.graph = tf.Graph()
        self.saver = None
        self.sess = None
        self.summ = None
        self.name = 'hln'
        self.model_path = '.'
        self.log_path = '.'

        # configurations
        self.learning_rate = 0.01
        self.stop_level = 0.01
        self.N_DIGITS = 5
        self.N = 8 
        self.H = 256
        self.W = 256
        self.C = 3

        # internal state
        self.layer_counter = 0 # the input layer is excluded

        # replace default settings with arguments
        if 'learning_rate' in kwargs: # learning rate
            self.learning_rate = kwargs['learning_rate']
        if 'stop_level' in kwargs: # not used now
            self.stop_level = kwargs['stop_level']
        if 'N_DIGITS' in kwargs: # class number
            self.N_DIGITS = kwargs['N_DIGITS']
        if 'N' in kwargs: # batch size
            self.N = kwargs['N']
        if 'H' in kwargs: # image height to resize to be
            self.H = kwargs['H']
        if 'W' in kwargs: # image width to resize to be
            self.W = kwargs['W']
        if 'C' in kwargs: # image channel required to fill
            self.C = kwargs['C']
        if 'name' in kwargs: # name of this network
            self.name = kwargs['name']

        with self.graph.as_default():
            # data feed in and back
            self.x = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self.N, self.H, self.W, self.C]
            )
            self.y = tf.placeholder(
                    dtype=tf.int32,
                    shape=[self.N]
            )

            # containers for layers and parameters
            self.hlc = [] # hybrid conv operators
            self.uws = [] # unsupervised weights
            self.sws = [] # supervised weights
            self.sbs = [] # supervised biases
            self.uts = [] # unsupervised training ops
            self.layers = [] # layers
            self.energies = [] # unsupervised energies

    def add_hlconv_layer(
            self,
            kernel_size=[3,3],
            num_filter=32,
            **kwargs
    ):
        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']
        if 'num_filter' in kwargs:
            num_filter = kwargs['num_filter']

        assert len(kernel_size) == 2

        with self.graph.as_default():
            self.layer_counter += 1
            ksizes = [1, kernel_size[0], kernel_size[1], 1]
            n = num_filter

            if self.layer_counter == 1:
                layer_input = self.x
            else:
                layer_input = self.layers[-1]

            uw, sw = hl.hlconv_make_params(
                n=n,
                depth=layer_input.shape.as_list()[3],
                ksizes=ksizes
            )

            self.uws.append(uw)
            self.sws.append(sw)

            hlconv, utrain_op, energy = hl.hlconv(
                x=layer_input,
                uw=uw,
                sw=sw,
                ksizes=ksizes,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name='%s-L%d' % (self.name, self.layer_counter)
            )

            self.hlc.append(hlconv)
            self.uts.append(utrain_op)
            self.energies.append(energy)

            # add bias
            bias = tf.Variable(
                tf.random_uniform(
                    [hlconv.shape.as_list()[3]]
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

    def add_output_layer(self):
        # A hybrid learning convolution layer is required to
        # convert the tensor into the shape as [N_DIGITS,...]
        self.add_hlconv_layer([1,1], self.N_DIGITS)

        with self.graph.as_default():
            # using max-pooling to get the output vector
            ksizes = [
                    1,
                    self.layers[-1].shape.as_list()[1],
                    self.layers[-1].shape.as_list()[2],
                    1
            ]
            self.logits = tf.nn.max_pool(
                    self.layers[-1],
                    ksizes,
                    strides = ksizes,
                    padding = 'VALID'
            )
            self.logits = tf.reshape(
                    self.logits, 
                    [self.N, self.N_DIGITS]
                    )

    def complete_network(self):
        self.add_output_layer()

        with self.graph.as_default():
            # get predicted label and its probability
            self.pred = tf.argmax(self.logits, 1)
            self.prob = tf.nn.softmax(self.logits)

            # for training, convert from labels into one-hot vector
            self.onehot = tf.one_hot(
                    tf.cast(self.y, tf.int32),
                    self.N_DIGITS,
                    1,
                    0
            )

            # training loss
            #self.loss = tf.reduce_sum(tf.square(self.y - feeds))
            self.loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=self.onehot,
                    logits=self.logits
            )
            
            # optimizer for supervised learning
            self.opt = tf.train.AdagradOptimizer(
                    learning_rate = self.learning_rate
            )

            # the only supervised training operator
            self.st = self.opt.minimize(self.loss)

    # generate graph with given network configuration
    # configuration json of a network can be like this:
    # {
    #       'name'  : 'net-001', # network name
    #       'hidden': [[3, 3, 16], [1, 1, 8]], # hidden layer design
    #       'x'     : [5, 5, 3], # input dimension with HWC format
    #       'y'     : [5] # output dimension with HWC format
    # }
    # all tensor shapes are described in NHWC format, which means
    # conf should be a json object containing 4 required
    # keyed elements: [name, hidden, x, y]
    def build_graph(self, conf):
        if type(conf)==unicode:
            pass
        elif type(conf)==str: # turn it into unicode first
            conf = conf.decode('utf8')
        else:
            assert False
        
        _conf = json.loads(conf, encoding='utf8')
        _keys = _conf.keys()

        assert list(_keys).__contains__(u'name')
        assert list(_keys).__contains__(u'hidden')
        assert list(_keys).__contains__(u'x')
        assert list(_keys).__contains__(u'y')

        # reset the network basic settings
        assert len(_conf[u'x']) == 3 # has to be format HWC

        self.H = _conf[u'x'][0]
        self.W = _conf[u'x'][1]
        self.C = _conf[u'x'][2]

        assert len(_conf[u'y']) == 1 # only classification task supported
        self.N_DIGITS = _conf[u'y'][0]

        assert type(_conf[u'name']) != 'unicode'
        self.name = _conf[u'name']

        # network body construction
        for _h in _conf[u'hidden']:
            self.add_hlconv_layer([_h[0],_h[1]], _h[2])

        self.complete_network()

    ################# DATA PREPARATION ##################
    # this will load all samples into memory for 
    # supervised training, test, or prediction
    # @path : path for data, it must be a dir containing 
    #         subdirs; for training or test, samples with 
    #         same labels have to be in same subdir.
    def offline_supervised(self, path):
        _subdirs = os.listdir(path)
        self.labels = [] # unicode strings
        self.sx_samples = []
        self.sy_samples = []
        
        for _subdir in _subdirs:
            self.labels.append(_subdir.decode('utf-8'))
        
        # to ensure an absolute mapping from id to label
        self.labels.sort()
        print 'please record down the list of labels:'

        # check datasets
        assert len(self.labels)==self.N_DIGITS
        
        for _i in xrange(len(self.labels)):
            _label = self.labels[_i]
            print(str(_i) + ' : ' + _label)
            _path = os.path.join(path, _label)
            _files = os.listdir(_path)
            for _file in _files:
                _im_path = os.path.join(
                        _path, 
                        _file.decode('utf-8')
                )
                _im = pi.open(_im_path)
                _im = _im.resize([self.W,self.H])
                _x = np.array(_im)
                # processing for non-3-channel images
                if len(_x.shape)==2: # gray image
                    _tmp = np.zeros([self.H,self.W,self.C])
                    for _ii in xrange(self.C):
                        _tmp[:,:,_ii] = _x
                    _x = _tmp
                elif len(_x.shape)==3 and _x.shape[2]!=self.C:
                    _tmp = np.zeros([self.H,self.W,self.C])
                    for _ii in xrange(self.C):
                        if _ii<_x.shape[2]:
                            _tmp[:,:,_ii] = _x[:,:,_ii]
                        else:
                            _tmp[:,:,_ii] = 0
                    _x = _tmp
                elif len(_x.shape)!=3:
                    raise NameError('unacceptable image format!')
                self.sx_samples.append(_x)
                self.sy_samples.append(_i)
                #pl.imshow(_im)
                #pl.show()
                #break
        # set up random generator(supervised)
        self.svol = len(self.sx_samples) # volume of samples
        self.sseq = np.arange(self.svol) # random sequence of samples
        self.sidx = 0 # current index in sequence
        
        # check if data is enough for a batch
        if self.svol < self.N:
            raise NameError('Samples not enough for a batch!')

        # construct supervised batch(x,y)
        self.sbatch_x = np.zeros(
                [self.N,self.H,self.W,self.C], 
                np.float32
        )
        self.sbatch_y = np.zeros([self.N], np.int32)

    # Generate a batch of samples(x,y) to feed the network
    def batch_supervised(self):
        if self.sidx+self.N > self.svol: # rest samples not enough for a batch
            np.random.shuffle(self.sseq)
            self.sidx = 0

        for _i in xrange(self.N):
            _idx = self.sseq[self.sidx + _i]
            self.sbatch_x[_i] = self.sx_samples[_idx]/255.0
            self.sbatch_y[_i] = self.sy_samples[_idx]

        self.sidx += self.N

    # this will load all samples into memory for 
    # unsupervised training and prediction
    # @path : ensure only images in current path
    def offline_unsupervised(self, path):
        self.ux_samples = [] 
        _files = os.listdir(path)
        
        for _file in _files:
            _im_path = os.path.join(
                    path, 
                    _file.decode('utf-8')
            )
            _im = pi.open(_im_path)
            _im = _im.resize([self.W,self.H])
            _x = np.array(_im)
            # processing for non-3-channel images
            if len(_x.shape)==2: # gray image
                _tmp = np.zeros([self.H,self.W,self.C])
                for _ii in xrange(self.C):
                    _tmp[:,:,_ii] = _x
                _x = _tmp
            elif len(_x.shape)==3 and _x.shape[2]!=self.C:
                _tmp = np.zeros([self.H,self.W,self.C])
                for _ii in xrange(self.C):
                    if _ii<_x.shape[2]:
                        _tmp[:,:,_ii] = _x[:,:,_ii]
                    else:
                        _tmp[:,:,_ii] = 0
                _x = _tmp
            elif len(_x.shape)!=3:
                raise NameError('unacceptable image format!')

            self.ux_samples.append(_x)
            #pl.imshow(_im)
            #pl.show()
            #break
        
        # set up random generator(unsupervised)
        self.uvol = len(self.ux_samples) # volume of samples
        self.useq = np.arange(self.uvol) # random sequence of samples
        self.uidx = 0 # current index in sequence
        
        # check if data is enough for a batch
        if self.uvol < self.N:
            raise NameError('Samples not enough for a batch!')

        # construct supervised batch(x)
        self.ubatch_x = np.zeros(
                [self.N,self.H,self.W,self.C], 
                np.float32
        )

    # Generate a batch of samples(x) to feed the network
    def batch_unsupervised(self):
        if self.uidx+self.N > self.uvol: # rest samples not enough for a batch
            np.random.shuffle(self.useq)
            self.uidx = 0

        for _i in xrange(self.N):
            _idx = self.useq[self.uidx + _i]
            self.ubatch_x[_i] = self.ux_samples[_idx]/255.0

        self.uidx += self.N

    # setting training model save dir and log dir
    def save_model_at(self, path):
        self.model_path = path

    def save_log_at(self, path):
        self.log_path = path

    # @nalt : number of alternation
    # @uepo : unsupervised iterations within an epoch
    # @sepo : supervised iterations within an epoch
    def train(self, nalt, uepo, sepo):
        print 'Training begin...'
        path = time.strftime('MODEL_%Y%m%d_%H%M%S')
        path = os.path.join(self.model_path, path)
        os.mkdir(path)
        print 'models saved in ', path

        with self.graph.as_default():
            if self.saver==None:
                self.saver = tf.train.Saver()
            if self.sess==None:
                self.sess = tf.Session()

            # initialization
            self.sess.run(tf.global_variables_initializer())
            
            for i in xrange(nalt):
                # run unsupervised training epoch
                for ui in xrange(uepo):
                    self.batch_unsupervised()
                    self.sess.run(
                            self.uts,
                            feed_dict={
                                self.x : self.ubatch_x
                            }
                    )
                for si in xrange(sepo):
                    self.batch_supervised()
                    self.sess.run(
                            self.st,
                            feed_dict={
                                self.x : self.sbatch_x,
                                self.y : self.sbatch_y
                            }
                    )
                # for each alternation, print the training losses
                # and the unsupervised energy level
                if __debug__:
                    print 'Training loss:', self.sess.run(
                            self.loss, 
                            feed_dict={
                                self.x : self.sbatch_x,
                                self.y : self.sbatch_y
                            })
                    print 'Unsupervised energies:', self.sess.run(
                            self.energies, 
                            feed_dict={
                                self.x : self.ubatch_x
                            })
                # every 100 steps to save a copy of the model
                if i%100 == 0 or i==nalt-1:
                    _model = os.path.join(path, 'hlconvnet')
                    self.saver.save(self.sess, _model, global_step=i)

    # @model : the model path
    def restore(self, model):
        with self.graph.as_default():
            if self.saver==None:
                self.saver = tf.train.Saver()
            if self.sess==None:
                self.sess = tf.Session()

            # restore variables to memory
            model_file = tf.train.latest_checkpoint(model)
            self.saver.restore(self.sess, model_file)
            print 'model loaded from files'

            # log down the graph
            path = time.strftime('LOG_%Y%m%d_%H%M%S')
            path = os.path.join(self.log_path, path)
            os.mkdir(path)
            print 'logs saved in ', path

            tf.summary.FileWriter(
                    path,
                    self.graph
            )

    # @data : the data path
    def predict(self, data):
        # sample volume = integer * batch_size
        assert len(os.listdir(data))%self.N==0
        n = len(os.listdir(data))/self.N

        # prepare data for running network
        self.offline_unsupervised(data)
        print 'test images loaded'

        with self.graph.as_default():
            # run the test and return output
            preds = []
            probs = []

            for _i in xrange(n):
                self.batch_unsupervised()
                pred, prob = self.sess.run(
                    [self.pred, self.prob],
                    feed_dict={
                        self.x : self.ubatch_x
                    })
                preds.append(pred)
                probs.append(prob)
            
            return (preds, probs)

