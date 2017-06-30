# test HLN on MNIST database
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [
       '\\usepackage{CJK}',
       r'\AtBeginDocument{\begin{CJK}{UTF8}{gbsn}}',
       r'\AtEndDocument{\end{CJK}}']
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import struct
import numpy as np
import sys
import matplotlib.pyplot as pl
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

def main():
	# reading MINST database
	# load training images
	filename = '../MNIST/train-images-idx3-ubyte'
	binfile = open(filename , 'rb')
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
	# loading training labels
	filename = '../MNIST/train-labels-idx1-ubyte'
	binfile = open(filename, 'rb')
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
        # prepare data for training and testing
        total = ims.shape[0]
	train = total/2
	test = total - train
        numlbl = 10
	# apply max-pooling
	ims1 = np.zeros([total,numRows*numColumns])
	ims1[:,:]=ims[0:total,:]
        del(ims)
	h = numRows
	w = numColumns
	[ims1,h,w] = max_pool(ims1,h,w,2,2)
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
        # training 
        snum = train*100
        sseq = np.random.randint(0,train,snum)
        serr = np.zeros(snum)
	for i in xrange(snum):
            v = ims1[sseq[i],:]
            f = np.zeros(numlbl) + 0.2
	    f[int(lbs[sseq[i]])] = 0.8
            serr[i] = net.drive_model(v, f)
            print(serr[i])
	# plot error figure
	serrfig = pl.figure()
	pl.plot(xrange(snum), serr, "r-")
	pl.title("Supervised Learning Error Curve")
	pl.xlabel("Iteration Time")
	pl.ylabel("Supervised Error")
	pl.ylim(0.0, 1.0)
	pl.draw()
	serrfig.savefig("serrfig")
	# check correctness
        predict = np.zeros(test)
	for i in xrange(test):
            (opt, err) = net.drive_model(ims1[train+i,:], [])
            predict[i] = np.argmax(opt)
	crt = 0
	for i in xrange(test):
		crt += bool(lbs[train+i]==predict[i])
	print(1.0*crt/test)
# execute the program
main()

# END OF FILE
