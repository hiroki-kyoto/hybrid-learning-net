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

def argsig(x):
    return -1.0*np.log(1.0/x-1)

def unify(x):
    #return np.ones(len(x))
    a = x.min()
    b = x.max()
    x = x + 1e-6
    t = a/(b-a)
    return b*t/x-t

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
            print "Network Model Parameter Not Well Set!"
            return
        self.layers = len(self.net_dim)
        if self.layers<3:
            print "Network Model Configuration Parameter is ILLEGAL!!!"
            print "Net layers should be no less than 3"
            return
        self.inputlayer = np.zeros([1, self.net_dim[0]])
        self.outputlayer = np.zeros([1, self.net_dim[self.layers-1]])
        self.olayerbias = np.random.rand(1, self.net_dim[self.layers-1])
        self.hiddenlayer = range(1, self.layers-1)
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
        # SOM connections
        self.som_conn = range(1, self.layers-1)
        self.bp_conn = range(1, self.layers)
        for i in range(1, self.layers):
            self.bp_conn[i-1] = np.random.rand(
                self.net_dim[i-1], self.net_dim[i])
            self.som_conn[i-1] = np.random.rand(
                self.net_dim[i-1], self.net_dim[i])
        # node error for each node
        self.node_error = range(1, self.layers)
        for i in range(1, self.layers):
            self.node_error[i-1] = np.zeros([1, self.net_dim[i]])

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
        self.inputlayer[0] = data/self.inputscale
        self.somlayer[0] = np.abs(self.som_conn[0]-self.inputlayer.T).sum(axis=0)
        mid = self.somlayer[0].argmin()
        self.somerror[0] = self.somlayer[0].min()
        self.somflag[0] = (self.somlayer[0].max()-self.someerror[0])/2.0
        self.somlayer[0] = unify(self.somlayer[0])
        # circle model updating
        decline = 1.0
        for i in range(0, self.som_rad):
            self.som_conn[0][
                :,
                (mid-i)%self.net_dim[1]
            ] += self.somerror[0]*self.som_eta*decline*(
                    self.inputlayer[0]-self.som_conn[0][
                        :,
                        (mid-i)%self.net_dim[1]
                    ]
                )
            if i>0:
                self.som_conn[0][
                    :,
                    (mid+i)%self.net_dim[1]
                ] += self.somerror[0]*self.som_eta*decline*(
                        self.inputlayer[0]-self.som_conn[0][
                            :,
                            (mid+i)%self.net_dim[1]
                        ]
                    )
            decline *= self.som_dec
        # feedforward computing
        self.hiddenlayer[0] = sigmoid(
            self.inputlayer.dot(
                self.bp_conn[0]
            ) * self.somlayer[0] + self.hlayerbias[0]
        )
        for i in range(2, self.layers-1):
            # run unsupervised learning first
            self.somlayer[i-1] = np.abs(
                self.som_conn[i-1]-self.hiddenlayer[i-2].T
                ).sum(axis=0)
            mid = self.somlayer[i-1].argmin()
            self.somerror[i-1] = self.somlayer[i-1].min()
            self.somflag[i-1] = (self.somlayer[i-1].max()-self.somerror[i-1])/2.0
            self.somlayer[i-1] = unify(self.somlayer[i-1])
            # circle model updating
            decline = 1.0
            for i in range(0, self.som_rad):
                self.som_conn[i-1][
                    :,
                    (mid-i)%self.net_dim[1]
                ] += self.somerror[i-1]*self.som_eta*decline*(
                    self.hiddenlayer[i-2]-self.som_conn[i-1][
                        :,
                        (mid-i)%self.net_dim[1]
                    ])
                if i>0:
                    self.som_conn[i-1][
                        :,
                        (mid+i)%self.net_dim[1]
                    ] += self.somerror[i-1]*self.som_eta*decline*(
                        self.hiddenlayer[i-2]-self.som_conn[i-1][
                            :,
                            (mid+i)%self.net_dim[1]
                    ])
                decline *= self.som_dec
            # update next hidden layer
            self.hiddenlayer[i-1] = sigmoid(
                self.hiddenlayer[i-2].dot(
                    self.bp_conn[i-1]
                )*self.somlayer[i-1] + self.hlayerbias[i-1])

        self.outputlayer = sigmoid(
            self.hiddenlayer[self.layers-3].dot(
                self.bp_conn[self.layers-2]
            ) + self.olayerbias
        )
        # check if to do feedback procedure
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
        eta = self.bp_eta*(1.0/(1.0+self.eta_coe*self.somflag[]))
        self.bp_conn[0] -= self.bp_eta*self.inputlayer.T.dot(
            self.node_error[0]
            )*self.somlayer[0]
        for i in range(1, self.layers-1):
            self.bp_conn[i] -= self.bp_eta*self.hiddenlayer[i-1].T.dot(
                node_error[i])
        # correcting bias of nodes
        self.olayerbias -= self.bp_eta*node_error[self.layers-2]
        for i in xrange(0, self.layers-2):
            self.hlayerbias[i] -= self.bp_eta*node_error[i]
        return error

# END OF FILE
