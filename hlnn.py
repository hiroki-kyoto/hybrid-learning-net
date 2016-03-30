# HLNN.py
# class of Hybrid Learning Neural Network

# class param definition
#   net_dim: dimensionality of whole net, like [2,100,2] (in circle)
#   som_eta: learning rate of SOM
#   som_rad: neighborhood radius
#   som_dec: decrease rate of correctness
#   =====================================
#   bp_eta: BPNN learning reate
#   bp_coe: BPNN learning rate coefficient[NOT USED]

import numpy as np


# helper methods
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def unify(x):
    s = x.sum()
    if s==0:
        return x
    else:
        return 1.0*x/s

class HLNN:
    def __init__(self):
        self.layers = 0
        self.net_dim = []
        self.som_eta = 0.3
        self.som_rad = 3
        self.som_dec = 0.2
        self.bp_eta = 0.3
        #self.bp_coe = 0.5
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
    #def set_bp_coe(self, bp_coe):
    #    self.bp_coe = bp_coe

    def build_model(self):
        # check if the dim is legal
        self.layers = len(self.net_dim)
        if self.layers<3:
            print "Network Model Configuration Parameter is ILLEGAL!!!"
            print "Net layers should be no less than 3"
            return
        self.inputlayer = np.zeros([1, self.net_dim[0]])
        self.somlayer = np.zeros([1, self.net_dim[1]])
        self.outputlayer = np.zeros([1, self.net_dim[self.layers-1]])
        self.olayerbias = np.random.rand(1, self.net_dim[self.layers-1])
        # hidden layers
        self.hiddenlayer = range(1, self.layers-1)
        self.hlayerbias = range(1, self.layers-1)
        for i in range(1, self.layers-1):
            self.hiddenlayer[i-1] = np.zeros([1, self.net_dim[i]])
            self.hlayerbias[i-1] = np.random.rand(1, self.net_dim[i])
        # build connections
        # SOM connections
        self.som_conn = np.random.rand(self.net_dim[0], self.net_dim[1])
        self.bp_conn = range(1, self.layers)
        for i in range(1, self.layers):
            self.bp_conn[i-1] = np.random.rand(
                self.net_dim[i-1], self.net_dim[i])

    # this method only work on single row training or predicting     
    # training or predicting is unified as only one API     
    def drive_model(self, data, feedback):         
        # check if model is built
        if self.layers<1:
            print "Error: Model is not built yet!"
            return
        # if input data has label then it is feedback training         
        # else it should be unsupervised learning on SOM model         
        if len(data) != self.net_dim[0]:
            print "Error: input data dimension is ILLEGAL!"             
            return         
        # run unsupervised learning first         
        self.inputlayer[0] = sigmoid(data)
        self.somlayer = np.abs(self.som_conn-self.inputlayer.T).sum(axis=0)
        mid = self.somlayer.argmax()
        self.somlayer = unify(self.somlayer)
        self.som_conn[:,mid] += self.som_eta*(
            self.inputlayer[0]-self.som_conn[:,mid])
        # circle model updating
        decline = 1.0
        for i in range(1, self.som_rad):  
            decline *= self.som_dec
            self.som_conn[:,(mid-i)%self.net_dim[1]] += \
            self.som_eta*decline*( \
                self.inputlayer[0]-self.som_conn[:,(mid-i)%self.net_dim[1]])
            self.som_conn[:,(mid+i)%self.net_dim[1]] += \
            self.som_eta*decline*( \
                self.inputlayer[0]-self.som_conn[:,(mid+i)%self.net_dim[1]])
        # feedforward computing
        self.hiddenlayer[0] = sigmoid(
            self.inputlayer.dot(
                self.bp_conn[0]
            ) * self.somlayer + self.hlayerbias[0]
        )
        for i in range(2, self.layers-1):
            self.hiddenlayer[i-1] = sigmoid(
                self.hiddenlayer[i-2].dot(
                    self.bp_conn[i-1]
                ) + self.hlayerbias[i-1]
            )
        self.outputlayer = sigmoid(
            self.hiddenlayer[self.layers-3].dot(
                self.bp_conn[self.layers-2]
            ) + self.olayerbias
        )
        # check if to do feedback procedure
        if feedback==[]:
            print "Model output finished!"
            return self.outputlayer
        # feedback part
        if len(feedback) != self.net_dim[self.layers-1]:
            print "feedback is of wrong dimension!"
            return
        # back-propagation
        # whole error:
        # convert feedback vector into signal vector
        feedback = sigmoid(feedback)
        error = (self.outputlayer-feedback)
        error = 0.5*(error*error).sum()
        print "error : ", error
        node_error = range(1, self.layers)
        for i in range(1, self.layers):
            node_error[i-1] = np.zeros([1, self.net_dim[i]])
        # back broadcast error
        node_error[self.layers-2] = (self.outputlayer-feedback)*(
            1-self.outputlayer)*self.outputlayer
        for i in range(self.layer-2, 0, -1):
            node_error[i-1] = (node_error[i].dot(self.bp_conn[i].T))*(
                1-self.hiddenlayer[i-1])*self.hiddenlayer[i-1]
        # correcting weight of connection
        self.bp_conn[0] += self.eta*self.inputlayer.T.dot(
            node_error[0])*self.somlayer
        for i in range(1, self.layers-1):
            self.bp_conn[i] += self.eta*self.hiddenlayer[i-1].T.dot(
                node_error[i])



