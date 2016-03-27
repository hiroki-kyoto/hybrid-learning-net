# HLNN.py
# class of Hybrid Learning Neural Network

# class param definition
#   net_dim: dimensionality of whole net, like [2,100,2] (in circle)
#   som_eta: learning rate of SOM
#   som_rad: neighborhood radius
#   som_dec: decrease rate of correctness
#   =====================================
#   bp_eta: BPNN learning reate
#   bp_coe: BPNN learning rate coefficient as cur*coe+last*(1-coe)

import numpy as np

class HLNN:
    def __init__(self):
        self.net_dim = []
        self.som_eta = 0.2
        self.som_rad = 3
        self.som_dec = 0.2
        self.bp_eta = 0.2
        self.bp_coe = 0.5
        print 'Creating new instance of HLNN model'
    @property
    def ready(self):
		if self.net_dim==[]:
			return False
		else:
			return True
    def set_net_dim(self, net_dim):
        self.net_dim = net_dim
    def set_som_eta(self, som_eta):
        self.som_eta = som_eta
    def set_som_rad(self, som_rad):
        self.som_rad = som_rad
    def set_som_dec(self, som_dec):
        self.som_dec = som_dec
    def set_bp_eta(self, bp_eta):
        self.bp_eta = bp_eta
    def set_bp_coe(self, bp_coe):
        self.bp_coe = bp_coe
    def build_model(self):
        # check if the dim is legal
        layers = len(net_dim)
        if layers<3:
            print "Network Model Configuration Parameter is ILLEGAL!!!"
            print "Net layers should be no less than 3"
            return
        self.inputlayer = np.zeros(self.dim[0])
        self.outputlayer = np.zeros(self.dim[layers-1])
        # hidden layers
        for i in range(1:layers-1):
            self.hiddenlayer[i-1] = np.zeros(self.net_dim[i])
        # build connections
        # SOM connections
        self.somnet = np.random.rand(self.net_dim[0], self.net_dim[1])
        for i in range(1:layers):
            self.bpnet[i-1] = np.random.rand(self.net_dim[i-1], self.net_dim[i])
    def drive_model(self, data, feedback):
        # if input data has label then it is feedback training
        # else it should be unsupervised learning on SOM model
        if len(data) != self.net_dim[0]:
            print "Error: input data dimension is ILLEGAL!"
            return
        # run unsupervised learning first
        self.inputlayer = data



