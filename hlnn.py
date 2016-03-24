# HLNN.py
# class of Hybrid Learning Neural Network

# class param definition
#   som_dim: dimensionality of SOM
#   som_eta: learning rate of SOM
#   som_rad: neighborhood radius
#   som_dec: decrease rate of correctness
#   bp_layer: layer number of BPNN
#   bp_dim: dimensionalities of all BPNN layers
#   bp_eta: BPNN learning reate
#   bp_coe: BPNN learning rate coefficient as cur*coe+last*(1-coe)


class HLNN:
    def __init__(self):
        # initialization of model settings
        self.som_dim = 0
        self.som_eta = 0
        self.som_rad = 0
        self.som_dec = 0
        self.bp_layer = 0
        self.bp_dim = 0
        self.bp_eta = 0
        self.bp_coe = 0
        print 'Creating new instance of HLNN model'
    @property
    def ready(self):
        setup = False
        if self.som_dim>0:
            if self.som_eta>0:
                if self.som_rad>0:
                    if self.som_dec>0:
                        if self.bp_layer>0:
                            if self.bp_dim>0:
                                if self.bp_eta>0:
                                    if self.bp.coe>0:
                                        setup = True
        return setup
    def set_model(self, som_dim, som_eta, som_rad, som_dec, bp_layer, bp_dim, bp_eta, bp_coe):
        self.som_dim = som_dim
        self.som_eta = som_eta
        self.som_rad = som_rad
        self.som_dec = som_dec
        self.bp_layer = bp_layer
        self.bp_dim = bp_dim
        self.bp_eta = bp_eta
        self.bp_coe = bp_coe
    def set_som_dim(self, som_dim):
        self.som_dim = som_dim
    def set_som_eta(self, som_eta):
	self.som_eta = som_eta
    def set_som_rad(self, som_rad):
	self.som_rad = som_rad
    def set_som_dec(self, som_dec):
	self.som_dec = som_dec
    def set_bp_layer(self, bp_layer):
	self.bp_layer = bp_layer
    def set_bp_dim(self, bp_dim):
	self.bp_dim = bp_dim
    def set_bp_eta(self, bp_eta):
	self.bp_eta = bp_eta
    def set_bp_coe(self, bp_coe):
	self.bp_coe = bp_coe
    def build_model(self):
        self.inputlayer = [];

