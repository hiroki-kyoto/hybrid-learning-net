import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [
       '\\usepackage{CJK}',
       r'\AtBeginDocument{\begin{CJK}{UTF8}{gbsn}}',
       r'\AtEndDocument{\end{CJK}}',
]
import matplotlib.pyplot as pl
import numpy as np
import sys
import os
import struct
#pl.style.use('ggplot')

# 1. neuron activation is probalistic.
# 2. sample learning is proactive.
# 3. unsupervised learning leads to reference learning(*)
# 4. small labeled data works too.
# 5. self-organizing map group
# 6. association(*)

# [NOTICE] to realize (3) and (6), we deal with this:
# (1) A compainied activation grows a strong connection
# (2) A decay ratio for strength of existing connections
# (3) It's not a end-to-end system>>> it's side-by-side
#     Input A and Output B all viewed as input signals
#     Output B connected to a hidden layer of net extending from A

# Proactive Probalistic Reference Net

# neural network is being constructed beyond layer based architecture
# neurons are arranged in a form of organization

def response_replaced(x, w):
    return np.dot(x,w)/(np.sqrt(np.dot(x,x))*np.sqrt(np.dot(w,w)))
def response(x, w):
    return 1.0 - np.max(np.abs(x-w))

def load_mnist(im_path, lb_path):
    # loading images
    binfile = open(im_path, 'rb')
    buf = binfile.read()
    index = 0
    magic,numImages,numRows,numColumns = \
    struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    if magic!=2051:
        raise NameError('MNIST TRAIN-IMAGE INCCORECT!')
    ims = np.zeros([numImages, numRows*numColumns])
    for i in range(numImages):
        ims[i,:] = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B');
    # loading labels
    binfile = open(lb_path, 'rb')
    buf = binfile.read()
    index = 0
    magic,numLabels = struct.unpack_from(
        '>II', 
        buf, 
        index
    )
    index += struct.calcsize('>II')
    if magic!=2049:
        raise NameError('MNIST TRAIN-LABEL INCORRECT!')
    lbs = np.zeros(numLabels)
    lbs[:] = struct.unpack_from(
        '>'+ str(numLabels) +'B', 
        buf, 
        index
    )
    return [ims, numRows, numColumns, lbs]


class PPRN:
    '''
    eta : learning rate for Probalistic Proactive Reference Net
    dim : neural network dimension
    hid : hidden layers
    som : parameters, self organizing map
    act : max activation for each som
    '''
    eta = 0
    dim = []
    hid = []
    som = []
    act = []
    
    def init(self, dim, eta):
        '''
        initial for the class
        dim : the dimension for PPRN
        eta : the learning rate
        '''
        self.eta = eta
        self.dim = dim
        self.hid = list(range(dim.shape[0]))
        self.som = list(range(dim.shape[0]))
        self.act = list(range(dim.shape[0]))
        
        for i in range(dim.shape[0]):
            self.hid[i] = np.zeros([self.dim[i,0], self.dim[i,1]])
            self.som[i] = np.random.rand(
                    self.dim[i, 0], 
                    self.dim[i, 1], 
                    self.dim[i, 2]
                    )
            self.act[i] = np.zeros(self.dim[i,0])
        print 'init done.'
        
    def train(self, x, y):
        '''
        training net with unlabeled data of [x]
        '''
        for i in range(self.dim.shape[0]):
            for j in range(self.dim[i,0]):
                for k in range(self.dim[i,1]):
                    if i==0:
                        self.hid[i][j,k] = response(x, self.som[i][j,k,:])
                    else:
                        self.hid[i][j,k] = response(self.act[i-1], \
                                self.som[i][j,k,:])
                self.act[i][j] = np.max(self.hid[i][j,:])
		# supervised learning only for last layer
		id_max = np.argmax(self.act[i])
		m = np.max(self.act[i])
		if i==self.dim.shape[0]-1 and y!=[]:
			while y!=id_max:
				# update parameters for the winner(supervised winner)
				if i==0: # first hidden layer
                			for k in range(self.dim[i,1]):
						self.som[i][y,k,:] += \
						self.eta*(self.hid[i][y,k]/m)*\
						(x-self.som[i][y,k,:])
				else:
					for k in range(self.dim[i,1]):
						self.som[i][y,k,:] += \
						self.eta*(self.hid[i][y,k]/m)*\
						(self.act[i-1]-self.som[i][y,k,:])
				# update parameters for the losers(supversied loser)
				if i==0:
					for j in range(self.dim[i,0]):
						for k in range(self.dim[i,1]):
							self.som[i][j,k,:] += \
							-self.eta*(self.hid[i][j,k])*\
							(self.act[i-1]-self.som[i][j,k,:])
				# update activation state with updated parameters
				for k in range(self.dim[i,1]):
					if i==0:
						self.hid[i][y,k] = response(x, self.som[i][y,k,:])
					else:
						self.hid[i][y,k] = response(self.act[i-1], \
						self.som[i][y,k,:])
				self.act[i][y] = np.max(self.hid[i][y,:])
				id_max = np.argmax(self.act[i])
		else:
			#if m > np.random.rand(1): # probalistic activation
			# update all patterns in som according to its response value
			if i == 0:
                        	for k in range(self.dim[i,1]):
                            		self.som[i][id_max,k,:] += \
                        		self.eta*(self.hid[i][id_max,k]/m)*\
					(x-self.som[i][id_max,k,:])
 			else:
                        	for k in range(self.dim[i,1]):
                            		self.som[i][id_max,k,:] += \
					self.eta*(self.hid[i][id_max,k]/m)*\
                            		(self.act[i-1]-self.som[i][id_max,k,:])

    def test(self, x):
        self.train(x, [])
        return np.argmax(self.hid[self.dim.shape[0]-1][:,0])

def main():
    print '======== init ========'
    net = PPRN()
    # reading MINST database
    # load training images
    train_im_path = '../MNIST/train-images-idx3-ubyte'
    train_lb_path = '../MNIST/train-labels-idx1-ubyte'
    test_im_path = '../MNIST/t10k-images-idx3-ubyte'
    test_lb_path = '../MNIST/t10k-labels-idx1-ubyte'
    # prepare data for training and testing
    print '===== load mnist ====='
    [ims_train, h, w, lbs_train] = load_mnist(train_im_path, train_lb_path)
    train_num = ims_train.shape[0]
    numlbl = 10
    # load test mnist data
    [ims_test, h, w, lbs_test] = load_mnist(test_im_path, test_lb_path)
    test_num = ims_test.shape[0]
    # apply a dataset
    #net.init(np.array([[5, 16, h*w], [numlbl, 8, 5]]), 0.1)
    net.init(np.array([[numlbl, 32, h*w]]), 0.1)
    for i in range(10000):
        net.train(ims_train[i,:]/255.0, [])
    print '=== training done ===='
    # test the model
    for i in range(20):
        print net.test(ims_train[i,:]/255.0), int(lbs_train[i])
    print '===== test done ======'
    
main()

