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
	
	if dim[1] != h*w:
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
	binfile = open(im_path, 'rb')
	buf = binfile.read()
	
	index = 0
	magic,numImages,numRows,numColumns = \
		struct.unpack_from('>IIII' , buf , index)
	index += struct.calcsize('>IIII')

	# for debug use only
	#numImages = 10
	
	if magic!=2051:
		raise NameError('MNIST TRAIN-IMAGE INCCORECT!')
	
	ims = np.zeros([numImages, numRows*numColumns])
	
	for i in range(numImages):
		ims[i,:] = struct.unpack_from('>784B', buf, index)
		index += struct.calcsize('>784B');
	
	binfile = open(lb_path, 'rb')
	buf = binfile.read()
	index = 0
	magic,numLabels = struct.unpack_from('>II', buf, index)
	index += struct.calcsize('>II')
	
	if magic!=2049:
		raise NameError('MNIST TRAIN-LABEL INCORRECT!')
	
	lbs = np.zeros(numLabels)
	lbs[:] = struct.unpack_from('>'+str(numLabels)+'B', buf, index)
	
	return [ims, numRows, numColumns, lbs]

def main():
	train_im_path = '../MNIST/train-images-idx3-ubyte'
	train_lb_path = '../MNIST/train-labels-idx1-ubyte'
	test_im_path = '../MNIST/t10k-images-idx3-ubyte'
	test_lb_path = '../MNIST/t10k-labels-idx1-ubyte'
	
	print('loading training images from MNIST...')
	
	[ims, h, w, lbs] = load_mnist(train_im_path, train_lb_path)
	train = ims.shape[0]
	numlbl = 10
	
	print('applying max-pooling to training images...')
	
	[ims1, h, w] = max_pool(ims, h, w, 2, 2)
	del(ims)
	
	print('loading testing images from MNIST...')
	
	[ims_test, h, w, lbs_test] = load_mnist(test_im_path, test_lb_path)
	test = ims_test.shape[0]
	
	print('applying max-pooling to testing images...')
	
	[ims1_test, h, w] = max_pool(ims_test, h, w, 2, 2)
	del(ims_test)
	
	print('building models.')
	
	if sys.argv[1]=="HLNN":
		net = HLNN()
		net_dim = [h*w, 30, 20, numlbl]
		net.set_net_dim(net_dim)
		net.set_scale(255.0, 1.0)
		net.set_bp_eta(0.8)
		net.set_som_rad(3)
		net.set_som_dec(1.0)
		net.set_som_eta(0.8)
		net.build_model()
			
	else:
		net = BPNN()
		net_dim = [h*w, 30, numlbl]
		net.set_net_dim(net_dim)
		net.set_scale(255.0, 1.0)
		net.set_bp_eta(0.8)
		net.build_model()

	print('setting up supervised training process.')
	
	sepn = 300 # supervised epoch number
	ratio = 1.0 # ratio of LABELED/TOTAL
	snum = train * sepn # supervised sample number
	
	# generate random sequence for supervised training
	sseq = np.random.randint(0,int(ratio*train),snum) 
	
	serr = np.zeros(sepn) # supervised error
	serr_save_path = 'serr.npy'
	tacc = np.zeros(sepn) # test accuracy
	tacc_save_path = 'tacc.npy'
	sspa = np.zeros([sepn,net.layers-2]) # layer sparsity
	sspa_save_path = 'sspa.npy'

	print('setting up unsupervised training process.')
	
	uepn = 1 # unsupervised epoch number
	unum = train * uepn # unsupervised sample number
	
	# generate random sequence for unsupervised training
	useq = np.random.randint(0,train,unum)
	
	# unsupervised error and its saving path
	uerr = np.zeros([uepn,net.layers-2])
	uerr_save_path = 'uerr.npy'
	
	for i in xrange(0, snum, 1):
		v = ims1[sseq[i],:]
		f = np.zeros(numlbl) + 0.2
		f[int(lbs[sseq[i]])] = 0.8
		
		serr[i/train] += net.train(v, f)
		sspa[i/train,:] += net.get_sparsity()
		
		if (i+1)%train==0: # epoch finished
			serr[i/train] /= train
			sspa[i/train,:] /= train
			
			crt = 0 # statistics for correct hits
			
			for ii in xrange(test): # check test accuracy
				opt = net.predict(ims1_test[ii,:])
				crt += bool(lbs_test[ii]==np.argmax(opt))
			tacc[i/train] = 1.0*crt/test
			
			prt_str = 'epoch#'
			prt_str += str((i+1)/train)
			prt_str += ': err='
			prt_str += str(serr[i/train])
			prt_str += '; acc='
			prt_str += str(tacc[i/train])
			prt_str += '; spa='
			prt_str += str(sspa[i/train,:])
			
			print(prt_str)
			
			# for HLNN, a novel training method is proposed:
			# if each epoch of the supervised training finished
			# with prominent alternation of model parameters,
			# then update the unsupervised model, or terminated
			# by the iteration limit.
			if net.type=='HLNN' and serr[i/train]<=1e-1:
				print('updating unsupervised model...')
				
				# initialize the unsupervised error
				for ii in xrange(0, uepn, 1):
					uerr[ii,:] = 0
				
				for ii in xrange(0, unum, 1):
					v = ims1[useq[ii],:]
					uerr[ii/train,:] += net.train(v,[])
					
					if (ii+1)%train==0: # epoch finished
						uerr[ii/train,:] /= train
						
						prt_str = 'epoch#'
						prt_str += str((ii+1)/train)
						prt_str += ':'
						prt_str += str(uerr[ii/train,:])
						
				print(prt_str)
	

	
	# save the unsupservised training states to files
	np.save(uerr_save_path, uerr)
	
	# save supervised training states to files
	np.save(serr_save_path, serr)
	np.save(tacc_save_path, tacc)
	np.save(sspa_save_path, sspa)

main()
# END OF FILE
