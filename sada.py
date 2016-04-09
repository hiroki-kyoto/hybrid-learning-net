# FILE NAME: hldnn.py
import numpy as np
import os
import sys
import math

# Autoencoder + BPNN classifier

class SADA:
	"SADA: SOM Aided Deep Autoencoder[connected with BP classifier]"
	def __init__(
		self, 
		in_dim, # input dimension
		ae_dim,  # autoencoder dimension
		classifier_dim,  # classifier dimension
		out_dim, # output dimension
		som_learning_rate, # SOM learning rate
		bp_learning_rate, # BP learning rate 
	):
		self.ind = in_dim
		self.aed = ae_dim
		self.cfd = classifier_dim
		self.otd = out_dim
		self.slr = som_learning_rate
		self.blr = bp_learning_rate
# INPUT AND OUTPUT LAYER
        self.il = np.zeros(self.ind)
        self.ol = np.zeros(self.otd)
# AUTO ENCODER BACKPROPAGATION LAYERS
		self.aebl = range(len(self.aed))
# AUTO ENCODER SOM LAYERS
		self.aesl = range(len(self.aed))
# AUTO ENCODER BP CONNECTIONS
		self.aebc = range(len(self.aed))
# AUTO ENCODER SOM CONNECTIONS
		self.aesc = range(len(self.aed))
# AE NET CONSTRUCTION
        for x in xrange(len(self.aed)):
			self.aebl[x] = np.zeros(self.aed[x])
			self.aesl[x] = np.zeros(self.aed[x])
        self.aebc[0] = np.random.rand(self.ind, self.aed[0])
        self.aesc[0] = np.random.rand(self.ind, self.aed[0]) 
        for x in xrange(1, len(aed)):
            self.aebc[x] = np.random.rand(self.aed[x-1], self.aed[x])
            self.aesc[x] = np.random.rand(self.aed[x-1], self.aed[x])
# CLASSIFIER BP LAYER
        self.cfbl = range(len(self.cfd))
# CLASSIFIER SOM LAYER
        self.cfsl = range(len(self.cfd))
# CLASSIFER NET CONSTRUCTION
        self.cfbc = range(len(self.cfd))
        self.cfsc = range(len(self.cfd))
        for x in xrange(len(self.cfd)):
            self.cfbl[x] = np.zeros(self.cfd[x])
            self.cfsl[x] = np.zeros(self.cfd[x])
        self.cfbc[0] = np.random.rand(self.aed[len(self.aed)-1], self.cfd[0])
        self.cfsc[0] = np.random.rand(self.aed[len(self.aed)-1], self.cfd[0])
        for x in xrange(1, len(self.cfd)):
            self.cfbc[x] = np.random.rand(self.cfd[x-1], self.cfd[x])
            self.cfsc[x] = np.random.rand(self.cfd[x-1], self.cfd[x])
# OUTPUT CONNECTION
        self.oc = np.random.rand(self.cfd[x], self.otd)
    
    def run(x, y):
        " y=[], predict, otherwise train the model "

