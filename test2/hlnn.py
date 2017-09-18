# HLNN.py
# class of Hybrid Learning Neural Network

# class param definition
#   net_dim: dimensionality of whole net, like [2,100,2]
#   som_eta: learning rate of SOM
#   som_rad: neighborhood radius
#   som_dec: decrease rate of correctness
#   =====================================
#   bp_eta: BPNN learning reate

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
        # check if the dim is legal
        if self.ready==False:
            print "Network Model Parameter Incomplete"
            return
        self.layers = len(self.net_dim)
        if self.layers<3:
            print "Network Model Configuration Error"
            print "Required: layer depth >= 3"
            return
        self.inputlayer = np.zeros([1, self.net_dim[0]])
        self.outputlayer = np.zeros([1, self.net_dim[self.layers-1]])
        self.olayerbias = np.random.rand(1, self.net_dim[self.layers-1])
        self.hiddenlayer = range(1, self.layers-1)
        self.sparsity = np.zeros(self.layers-2)
        # each hidden layer has a friend som layer 
        # to estimate its signal distribution
        self.somlayer = range(1, self.layers-1)
        self.somflag = np.zeros(len(self.somlayer))
        self.somerror = np.zeros(len(self.somlayer))
        self.hlayerbias = range(1, self.layers-1)
        for i in range(1, self.layers-1):
            self.hiddenlayer[i-1] = np.zeros([1, self.net_dim[i]])
            self.somlayer[i-1] = np.zeros([1, self.net_dim[i]])
            self.hlayerbias[i-1] = np.random.rand(1, self.net_dim[i])
        # build connections
        self.som_conn = range(1, self.layers-1)
        self.bp_conn = range(1, self.layers)
        for i in range(1, self.layers):
            self.bp_conn[i-1] = np.random.rand(
                self.net_dim[i-1], self.net_dim[i])
        for i in xrange(self.layers-2):
            self.som_conn[i] = np.random.rand(
                self.net_dim[i], self.net_dim[i+1])
        # node error for each node
        self.node_error = range(1, self.layers)
        for i in range(1, self.layers):
            self.node_error[i-1] = np.zeros([1, self.net_dim[i]])
    
    def feed(self, x):
        self.inputlayer[0] = data/self.inputscale
    
    def forward(self, il):
        # flow over som
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
            t = np.dot(self.inputlayer, self.bp_conn[il])
        else:
            t = np.dot(self.hiddenlayer[il-1], self.bp_conn[il])
        t = self.somlayer[il]+self.hlayerbias[il]
        self.hiddenlayer[il] = sigmoid(t)
        # layer sparsity
        t = filter(lambda x:x>1e-3, self.hiddenlayer[il][0,:])
        self.sparsity[il] = 1.0*len(t)/len(self.hiddenlayer[il][0,:])
    
    def update(self, il):
        # update som connections
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
    
    def drive_model(self, data, feedback): 
        if self.layers<1:
            print "Error: Model is not built yet!"
            return
        
        if len(data) != self.net_dim[0]:
            print "Error: input data dimension is ILLEGAL!"
            return

        self.feed(data)
        for i range(0,len(self.net_dim)):
            self.forward(0)
        
        if feedback == []:
            t = self.som_conn[0]-self.inputlayer.T
            t = np.abs(t).sum(axis=0)/self.net_dim[0]
            self.somlayer[0] = t
            mid = self.somlayer[0].argmin()
            self.somerror[0] = self.somlayer[0].min()
            self.somflag[0] = self.somlayer[0].max()-self.somerror[0]
            self.somlayer[0] = unify(self.somlayer[0])

            for i in range(-self.som_rad, self.som_rad+1):
                t = self.som_conn[0][:,(mid+i)%self.net_dim[1]]
                t = self.inputlayer[0] - t
                decay = np.power(som_dec, np.abs(i))
                t = self.somerror[0]*self.som_eta*decay*t
                self.som_conn[0][:,(mid+i)%self.net_dim[1]] += t
#### notice : in new version, we need to implement multi-layer unsupervised learning #####
        # indent to be fixed
        # feedforward computing
        self.hiddenlayer[0] = sigmoid(
            self.inputlayer.dot(
                self.bp_conn[0]
            ) * self.somlayer[0] + self.hlayerbias[0]
        )
        self.sparsity[0] = 1.0*len(filter(lambda x:x>1e-3,self.hiddenlayer[0][0,:]))/len(self.hiddenlayer[0][0,:])
        for i in range(2, self.layers-1):
            # run unsupervised learning first
            self.somlayer[i-1] = np.abs(
                self.som_conn[i-1]-self.hiddenlayer[i-2].T
                ).sum(axis=0)/self.net_dim[i-1]
            mid = self.somlayer[i-1].argmin()
            self.somerror[i-1] = self.somlayer[i-1].min()
            self.somflag[i-1] = self.somlayer[i-1].max()-self.somerror[i-1]
            self.somlayer[i-1] = unify(self.somlayer[i-1])
            # circle model updating
            decline = 1.0
            for k in range(0, self.som_rad):
                self.som_conn[i-1][:,(mid-k)%self.net_dim[i]] += \
                self.somerror[i-1]*self.som_eta*decline*(
                    self.hiddenlayer[i-2][0][:]-self.som_conn[i-1][:,(mid-k)%self.net_dim[i]])
                if k>0:
                    self.som_conn[i-1][:,(mid+k)%self.net_dim[i]] += \
                    self.somerror[i-1]*self.som_eta*decline*(
                        self.hiddenlayer[i-2][0][:]-self.som_conn[i-1][:,(mid+k)%self.net_dim[i]])
                decline *= self.som_dec
            # update next hidden layer
            self.hiddenlayer[i-1] = sigmoid(
                self.hiddenlayer[i-2].dot(
                    self.bp_conn[i-1]
                )*self.somlayer[i-1] + self.hlayerbias[i-1])
            self.sparsity[i-1] = 1.0*len(filter(lambda x:x>1e-3,self.hiddenlayer[i-1][0,:]))/len(self.hiddenlayer[i-1][0,:])
        self.outputlayer = sigmoid(
            self.hiddenlayer[self.layers-3].dot(
                self.bp_conn[self.layers-2]
            ) + self.olayerbias
        )

        if feedback==[]:
            #print "output: ", self.outputlayer
            return (self.outputlayer*self.outputscale, self.somflag)
        # feedback part
        if len(feedback) != self.net_dim[self.layers-1]:
            print "feedback is of wrong dimension!"
            return
        # back-propagation
        # whole error:
        # convert feedback vector into signal vector
        # print self.outputlayer*self.outputscale, feedback
        feedback = feedback/self.outputscale
        error = self.outputlayer-feedback
        error = 0.5*(error*error).sum()
        # back broadcast error
        self.node_error[self.layers-2] = (
            self.outputlayer-feedback
            )*(
            1-self.outputlayer
            )*self.outputlayer
        # THE LAST HIDDEN LAYER
        self.node_error[self.layers-3] = (
            self.node_error[self.layers-2].dot(
                self.bp_conn[self.layers-2].T
                )*(
                1-self.hiddenlayer[self.layers-3]
                )*self.hiddenlayer[self.layers-3]
            )
        for i in range(self.layers-3, 0, -1):
            self.node_error[i-1] = (
                (self.somerror[i]*self.node_error[i]).dot(self.bp_conn[i].T)
                )*(
                1-self.hiddenlayer[i-1]
                )*self.hiddenlayer[i-1]
        # correcting weight of connection
        eta = self.bp_eta*(1.0/(1.0+self.eta_coe*self.somflag[0]))
        self.bp_conn[0] -= eta*self.inputlayer.T.dot(
            self.node_error[0]
            )*self.somlayer[0]
        self.hlayerbias[0] -= eta*self.node_error[0]
        for i in range(1, self.layers-2):
            eta = self.bp_eta*(1.0/(1.0+self.eta_coe*self.somflag[i]))
            self.bp_conn[i] -= eta*self.hiddenlayer[i-1].T.dot(
                self.node_error[i])*self.somlayer[i]
            self.hlayerbias[i] -= eta*self.node_error[i]
        # for the output layer
        eta = self.bp_eta
        self.bp_conn[self.layers-2] -= eta*self.hiddenlayer[
            self.layers-3].T.dot(
                self.node_error[self.layers-2]
            )
        # correcting bias of output layer
        self.olayerbias -= eta*self.node_error[self.layers-2]
        # return net error
        return error
    def get_sparsity(self):
        return self.sparsity

# END OF FILE
