# test HLN on MNIST database
import numpy as np
import sys
import os
import struct
from hlnn import HLNN
from bpnn import BPNN

# using HLN to classify these digits
# MAX-OUT OPERATOR
def max_pool(ims, h, w, ph, pw):
    dim = ims.shape
    if dim[1] <> h*w:
        raise NameError('image dimension error')
    nh = (h-1)/ph + 1
    nw = (w-1)/pw + 1
    nims = np.zeros([dim[0], nh*nw])
    pool = np.zeros(ph*pw)
    for i in range(dim[0]):
        for j in range(nh):
            for k in range(nw):
                for s in range(ph):
                    for t in range(pw):
                        if j*ph+s>=h:
                            pool[s*pw+t] = 0
                        elif k*pw+t>=w:
                            pool[s*pw+t] = 0
                        else:
                            pool[s*pw+t] = ims[i,(j*ph+s)*w+k*pw+t]
                nims[i,j*nw+k] = np.max(pool)
    return [nims, nh, nw]

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

def main():
	# reading MINST database
	# load training images
	train_im_path = '../MNIST/train-images-idx3-ubyte'
	train_lb_path = '../MNIST/train-labels-idx1-ubyte'
	test_im_path = '../MNIST/t10k-images-idx3-ubyte'
	test_lb_path = '../MNIST/t10k-labels-idx1-ubyte'
	# prepare data for training and testing
	print('loading images and labels from MNIST...')
	[ims, h, w, lbs] = load_mnist(train_im_path, train_lb_path)
	train = ims.shape[0]
        # number of labels from 0 to 9
        numlbl = 10
	# apply max-pooling
	#print('applying max-pooling...')
	#[ims1, h, w] = max_pool(ims,h,w,2,2)
        ims1 = ims
	del(ims)
        # load test mnist data
        [ims_test, h, w, lbs_test] = load_mnist(test_im_path, test_lb_path)
	test = ims_test.shape[0]
	#[ims1_test, h, w] = max_pool(ims_test, h, w, 2, 2)
        ims1_test = ims_test
        del(ims_test)
        # deep HLN => DHLN
	if sys.argv[1]=="HLNN":
		net = HLNN()
	else:
		net = BPNN()
        # build net model
	net_dim = [h*w, 30, numlbl]
	net.set_net_dim(net_dim)
	net.set_scale(255.0, 1.0)
	net.set_bp_eta(0.8)
	net.set_som_rad(3)
	net.set_som_dec(1.0)
	net.set_som_eta(0.8)
	net.build_model()
        # supervised training 
        epn = 300
        labeled_ratio = 0.7
        strain = int(np.floor(train * labeled_ratio))
        snum = strain * epn
        sseq = np.random.randint(0,strain,snum) 
        k = 0
        epi = 0
        err = 0.0
        serr = np.zeros(epn) # supervised learning error
        tacc = np.zeros(epn) # test accuracy
        spar = np.zeros([net.layers-2, epn]) # neural net sparsity
        # run unsupervised training for HLN
        uepn = 1
        unum = train * uepn
        useq = np.random.randint(0,train,unum)
        uerr = 0.0
        for i in xrange(unum):
            v = ims1[useq[i],:]
            [_f, _err] = net.drive_model(v, [])
            uerr += _err
            if i%train==0:
                print i/train, ':', uerr/train
                uerr = 0
        # supervised learning
	for i in xrange(snum):
            v = ims1[sseq[i],:]
            f = np.zeros(numlbl) + 0.2
	    f[int(lbs[sseq[i]])] = 0.8
            err += net.drive_model(v, f)
            spar[:,epi] = spar[:,epi] + net.get_sparsity()
            k += 1
            if k==strain:
                serr[epi] = err/strain
                # run test dataset
                crt = 0
                for ii in xrange(test):
                    (opt, _err) = net.drive_model(ims1_test[ii,:], [])
                    crt += bool(lbs_test[ii]==np.argmax(opt))
                tacc[epi] = 1.0*crt/test
                spar[:,epi] = spar[:,epi]/strain
                print epi,'\t',serr[epi],'\t',tacc[epi],'\t',spar[:,epi]
                # update training state
                k = 0
                epi += 1
                err = 0
main()
# END OF FILE
