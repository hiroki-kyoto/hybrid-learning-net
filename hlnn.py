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
    def set_model(self, net_dim):
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
        self.inputlayer = np.arange(self.dim[0]);





