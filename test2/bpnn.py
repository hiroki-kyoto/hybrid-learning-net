# HLNN.py
# class of Hybrid Learning Neural Network

# class param definition
#   net_dim: dimensionality of whole net, like [2,100,2] (in circle)
#   =====================================
#   bp_eta: BPNN learning reate
#   bp_coe: BPNN learning rate coefficient[NOT USED]

import numpy as np


# helper methods
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def argsig(x):
    return -1.0*np.log(1.0/x-1)

class BPNN:
    def __init__(self):
        self.layers = 0
        self.net_dim = []
        self.bp_eta = 0.3
        self.inputscale = 1.0
        self.outputscale = 1.0

    @property
    def ready(self):
		if self.net_dim==[]:
			return False
		else:
			return True

    def set_net_dim(self, net_dim):
        self.net_dim = net_dim
    
    def set_bp_eta(self, bp_eta):
        self.bp_eta = bp_eta
    
    def set_scale(self, inputscale, outputscale):
        self.inputscale = inputscale
        self.outputscale = outputscale

    def build_model(self):
        if self.ready==False:
            raise NameError("network model incomplete")
        self.layers = len(self.net_dim)
        if self.layers<3:
            print "Network Model Configuration Parameter is ILLEGAL!!!"
            print "Net layers should be no less than 3"
            return
        self.inputlayer = np.zeros([1, self.net_dim[0]])
        self.outputlayer = np.zeros([1, self.net_dim[self.layers-1]])
        self.olayerbias = np.random.rand(1, self.net_dim[self.layers-1])
        # hidden layers
        self.hiddenlayer = range(1, self.layers-1)
        self.sparsity = np.zeros(self.layers-2)
        self.hlayerbias = range(1, self.layers-1)
        for i in range(1, self.layers-1):
            self.hiddenlayer[i-1] = np.zeros([1, self.net_dim[i]])
            self.hlayerbias[i-1] = np.random.rand(1, self.net_dim[i])
        # build connections
        self.bp_conn = range(1, self.layers)
        for i in range(1, self.layers):
            self.bp_conn[i-1] = np.random.rand(
                self.net_dim[i-1], self.net_dim[i])
    def drive_model(self, data, feedback):
        # check if model is built
        if self.layers<1:
            print "Error: Model is not built yet!"
            return
        if len(data) != self.net_dim[0]:
            print "Error: input data dimension is ILLEGAL!"             
            return         
        # unifying inputs
        self.inputlayer[0] = data/self.inputscale
        # feedforward computing
        self.hiddenlayer[0] = sigmoid(
            self.inputlayer.dot(
                self.bp_conn[0]
            ) + self.hlayerbias[0]
        )
        self.sparsity[0] = 1.0*len(filter(lambda x:x>1e-3,self.hiddenlayer[0][0,:]))/len(self.hiddenlayer[0][0,:])
        for i in range(2, self.layers-1):
            self.hiddenlayer[i-1] = sigmoid(
                self.hiddenlayer[i-2].dot(
                    self.bp_conn[i-1]
                ) + self.hlayerbias[i-1]
            )
            self.sparsity[i-1] = 1.0*len(filter(lambda x:x>1e-3,self.hiddenlayer[i-1][0,:]))/len(self.hiddenlayer[i-1][0,:])
        self.outputlayer = sigmoid(
            self.hiddenlayer[self.layers-3].dot(
                self.bp_conn[self.layers-2]
            ) + self.olayerbias
        )
        if feedback==[]:
            return (self.outputlayer*self.outputscale, 0) 
        if len(feedback) != self.net_dim[self.layers-1]:
            print "feedback is of wrong dimension!"
            return
        feedback = feedback/self.outputscale
        error = self.outputlayer-feedback
        error = 0.5*(error*error).sum()
        node_error = range(1, self.layers)
        for i in range(1, self.layers):
            node_error[i-1] = np.zeros([1, self.net_dim[i]])
        # back broadcast error
        node_error[self.layers-2] = (self.outputlayer-feedback)*(
            1-self.outputlayer)*self.outputlayer
        for i in range(self.layers-2, 0, -1):
            node_error[i-1] = (node_error[i].dot(self.bp_conn[i].T))*(
                1-self.hiddenlayer[i-1])*self.hiddenlayer[i-1]
        # correcting weight of connection
        self.bp_conn[0] -= self.bp_eta*self.inputlayer.T.dot(node_error[0])
        for i in range(1, self.layers-1):
            self.bp_conn[i] -= self.bp_eta*self.hiddenlayer[i-1].T.dot(node_error[i])
        # correcting bias of nodes
        self.olayerbias -= self.bp_eta*node_error[self.layers-2]
        for i in xrange(0, self.layers-2):
            self.hlayerbias[i] -= self.bp_eta*node_error[i]
        return error
    def get_sparsity(self):
        return self.sparsity


# END OF FILE
