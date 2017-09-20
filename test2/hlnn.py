# HLNN.py
# class of Hybrid Learning Neural Network

# class param definition
#   net_dim: dimensionality of whole net, like [2,100,2]
#   som_eta: learning rate of SOM
#   som_rad: neighborhood radius
#   som_dec: decrease rate of correctness
#   =====================================
#   bp_eta: BPNN learning reate
#   flag:   0 unsupservised; 
#           1 supservised; 
#           2 hybrid;

import numpy as np

# helper methods
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def unify(x):
    #return np.ones(len(x))
    x = x + 1e-5
    a = x.min()
    b = x.max()
    t = a/(b-a+1e-5)
    return b*t/x-t
#def unify(x):
#    x =  
class HLNN:
    def __init__(self):
        self.layers = 0
        self.net_dim = []
        self.som_eta = 0.2
        self.som_rad = 2
        self.som_dec = 0.2
        self.bp_eta = 0.3
        self.eta_coe = 2.0
        self.inputscale = 1.0
        self.outputscale = 1.0
        #self.bp_coe = 0.5
        #print 'Creating new instance of HLNN model'

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
    def set_scale(self, inputscale, outputscale):
        self.inputscale = inputscale
        self.outputscale = outputscale

    def build_model(self):

        if self.ready==False:
            raise NameError("model incomplete")

        self.layers = len(self.net_dim)
        
        if self.layers<3:
            raise NameError("at least 3 layers required")

        self.inputlayer = np.zeros([1, self.net_dim[0]])
        self.outputlayer = np.zeros([1, self.net_dim[self.layers-1]])
        self.olayerbias = np.random.rand(1, self.net_dim[self.layers-1])
        self.hiddenlayer = range(1, self.layers-1)
        self.sparsity = np.zeros(self.layers-2)
        
        self.somlayer = range(1, self.layers-1)
        self.somflag = np.zeros(len(self.somlayer))
        self.somerror = np.zeros(len(self.somlayer))
        self.hlayerbias = range(1, self.layers-1)
        
        for i in range(1, self.layers-1):
            self.hiddenlayer[i-1] = np.zeros([1, self.net_dim[i]])
            self.somlayer[i-1] = np.zeros([1, self.net_dim[i]])
            self.hlayerbias[i-1] = np.random.rand(1, self.net_dim[i])

        self.som_conn = range(1, self.layers-1)
        self.bp_conn = range(1, self.layers)
        
        for i in range(1, self.layers):
            self.bp_conn[i-1] = np.random.rand(
                self.net_dim[i-1], self.net_dim[i])
        
        for i in xrange(self.layers-2):
            self.som_conn[i] = np.random.rand(
                self.net_dim[i], self.net_dim[i+1])

        self.node_error = range(1, self.layers)
        
        for i in range(1, self.layers):
            self.node_error[i-1] = np.zeros([1, self.net_dim[i]])
    
    def feed(self, x):
        self.inputlayer[0] = data/self.inputscale
    
    def forward(self, il):
        
        # flow over som
        if il<self.layers-2:
            if il==0:
                t = self.som_conn[il] - self.inputlayer.T
            else:
                t = self.som_conn[il] - self.hiddenlayer[il-1].T
            
            t = np.abs(t).sum(axis=0)/self.net_dim[il]
            self.somerror[il] = t.min()
            self.somflag[il] = t.max()-self.somerror[il]
            self.somlayer[il] = unify(t)
        
        # flow over fcn
        if il==0:
            t = self.inputlayer
            t = np.dot(t, self.bp_conn[il])
            t = t*self.somlayer[il] + self.hlayerbias[il]
            self.hiddenlayer[il] = sigmoid(t)
            t = filter(lambda x:x>1e-3, self.hiddenlayer[il][0,:])
            self.sparsity[il] = 1.0*len(t)/len(self.hiddenlayer[il][0,:])
        elif il==self.layers-2:
            t = self.hiddenlayer[il-1]
            t = np.dot(t, self.bp_conn[il])
            t = t + self.olayerbias
            self.outputlayer = sigmoid(t)
        else:
            t = self.hiddenlayer[il-1]
            t = np.dot(t, self.bp_conn[il])
            t = t*self.somlayer[il] + self.hlayerbias[il]
            self.hiddenlayer[il] = sigmoid(t)
            t = filter(lambda x:x>1e-3, self.hiddenlayer[il][0,:])
            self.sparsity[il] = 1.0*len(t)/len(self.hiddenlayer[il][0,:])
    
    def update(self, il, flag):
        
        # update som connections
        if flag==0 || flag==2:

            if il<self.layers-2:
                mid = self.somlayer[il].argmin()
                
                for i in range(-self.som_rad, self.som_rad+1):
                    t = self.som_conn[il][:,(mid+i)%self.net_dim[il+1]]
                    
                    if il==0:
                        t = self.inputlayer[0] - t
                    else:
                        t = self.hiddenlayer[il-1][0][:] - t
                    
                    decay = np.power(som_dec, np.abs(i))
                    t = self.somerror[il]*self.som_eta*decay*t
                    self.som_conn[il][:,(mid+i)%self.net_dim[il+1]] += t
        
        # update fcn connections
        if flag==1 || flag==2:
            
            if il==0:
                t = self.inputlayer.T
            else:
                t = self.hiddenlayer[il-1].T
            
            t = np.dot(t,self.node_error[il])

            if il==self.layers-2:
                self.bp_conn[il] -= self.bp_eta*t
                self.olayerbias -= self.bp_eta*self.node_error[il]
            else:
                self.bp_conn[il] -= self.bp_eta*t*self.somlayer[il]
                self.hlayerbias[il] -= self.bp_eta*self.node_error[il]

    def backward(self, il, y):
        
        if il==self.layers-2:
            t = self.outputlayer - y
            t = t*self.outputlayer*(1-self.outputlayer)
            self.node_error[il] = t
        
        if il==self.layers-3:
            t = self.bp_conn[il+1].T
            t = np.dot(self.node_error[il+1], t)
            t = t*self.hiddenlayer[il]*(1-self.hiddenlayer[il])
            self.node_error[il] = t
        
        else:
            t = self.somerror[il+1]*self.node_error[il+1]
            t = np.dot(t, self.bp_conn[il+1].T)
            t = t*self.hiddenlayer[il]*(1-self.hiddenlayer[il])
            self.node_error[il] = t

    def error(self, y):
        y = y/self.outputscale
        error = self.outputlayer-y
        error = 0.5*(error*error).sum()
        
        return error

    def check_model(self):
        
        if self.layers<1:
            raise NameError("model incomplete")
    
    def check_data(self, x):
        
        if len(x) != self.net_dim[0]:
            raise NameError("bad input dimension")
    
    def check_feedback(self, y):
        
        if len(y) != self.net_dim[self.layers-1]:
            raise NameError("bad feedback dimension")

    def train(self, data, feedback):
        self.check_model()
        self.check_data(data)
        self.feed(data)
        
        for i range(0, self.layers-1):
            self.forward(i)

        if feedback==[]:

            for i range(self.layer-2, -1, -1):
                self.update(i, 0)
            
            t = self.outputlayer*self.outputscale
            
            return self.somflag
        
        else: 
            self.check_feedback(feedback)
            error = self.error()
            
            for i range(self.layers-2, -1, -1):
                self.backward(i, feedback)
            
            for i range(self.layer-2, -1, -1):
                self.update(i, 1)
            
            return error

    def get_sparsity(self):
        
        return self.sparsity

    def predict(self, data):
        self.train(data, [])
        
        return self.outputlayer*self.outputscale


# END OF FILE
