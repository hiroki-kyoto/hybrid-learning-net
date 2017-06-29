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
	# apply max-pooling
	ims1 = np.zeros([10,numRows*numColumns])
	ims1[:,:]=ims[0:10,:]
	h = numRows
	w = numColumns
	[ims1,h,w] = max_pool(ims1,h,w,2,2)

	for i in range(10):
		im = np.array(ims1[i,:])
		im = im.reshape(h,w)
		fig = plt.subplot(1,10,i+1)
		fig.imshow(im , cmap='gray')
		fig.axis('off')
		plt.title(str(int(lbs[i])))
	plt.show()
	return
	# deep HLN => DHLN

	#print 'HLNN Instance Running Test'
	if sys.argv[1]=="HLNN":
		net = HLNN()
	else:
		net = BPNN()
	###############################################
	# generate samples of test
	###############################################
	total = 400
	train = total/2
	test = total - train
	d = np.array([2.3, 3.0, 2.5])
	r = np.array([6.1, 17.3, 28.2])
	c = np.array([
		[33, 33],
		[34, 31],
		[36, 32]
		])
	# plot them out
	data = range(3)
	for i in xrange(0, len(r)):
		rou = r[i]-d[i] + 2*d[i]*np.random.rand(total)
		theta = np.random.rand(total)*2*np.pi
		data[i] = np.array([
			c[i, 0] + rou*np.cos(theta),
			c[i, 1] + rou*np.sin(theta)
			])
	# draw figure to show points of topologies
	fig1 = pl.figure()
	pl.plot(data[0][0], data[0][1], "*")
	pl.plot(data[1][0], data[1][1], ".")
	pl.plot(data[2][0], data[2][1], "x")
	pl.xlabel('x')
	pl.ylabel('y')
	pl.title('Classification of Random Topologies')
	#pl.show()
	pl.draw()
	fig1.savefig("test2")
	##################################################
	# applying model to test samples
	##################################################
	net_dim = [2, 1000, len(r)]
	net.set_net_dim(net_dim);
	net.set_scale(100.0, 1.0)
	net.set_bp_eta(0.8)
	net.set_som_rad(3)
	net.set_som_dec(1.0)
	net.set_som_eta(0.8)
	net.build_model()
	# prepare unsupervised and supervised data
	unum = 0 # unsupervised learning
	snum = 5000  # supervised learning
	ulbl = np.random.randint(0, len(r), size=unum)
	useq = np.random.randint(0, train, size=unum)
	uerr = np.zeros([len(net_dim)-2, unum])
	for i in xrange(unum):
		v = np.array([
			data[ulbl[i]][0][useq[i]],
			data[ulbl[i]][1][useq[i]]
			])
		(uopt, uerr[:,i]) = net.drive_model(v, []) # feedback=[] means no feedback
	# plot unsupervised error level
	uerrfig = pl.figure()
	pl.plot(xrange(unum), uerr[0], "-")
	pl.xlabel("Iteration Time")
	pl.ylabel("SOM State Flag")
	pl.title("Unsupervised Learning (SOM)")
	pl.ylim(0.0, 1.0)
	#pl.show()
	pl.draw()
	uerrfig.savefig("uerrfig")
	#print "Model unsupervised learning process done!"
	# supervised learning
	slbl = np.random.randint(0, len(r), size=snum)
	sseq = np.random.randint(0, train, size=snum)
	serr = np.zeros(snum)
	for i in xrange(snum):
		v = np.array([
			data[slbl[i]][0][sseq[i]],
			data[slbl[i]][1][sseq[i]]
			])
		f = np.zeros(len(r)) + 0.2
		f[slbl[i]] = 0.8
		serr[i] = net.drive_model(v, f)
	# plot error figure
	serrfig = pl.figure()
	pl.plot(xrange(snum), serr, "-")
	pl.title("Supervised Learning Error Curve")
	pl.xlabel("Iteration Time")
	pl.ylabel("Supervised Error")
	pl.ylim(0.0, 1.0)
	#pl.show()
	pl.draw()
	serrfig.savefig("serrfig")
	# check correctness
	cnum = 800
	clbl = np.random.randint(0, len(r), size=cnum)
	cseq = np.random.randint(train, train+test, size=cnum)
	plbl = np.random.randint(0, len(r), size=cnum)
	for i in xrange(cnum):
		v = np.array([
			data[clbl[i]][0][cseq[i]],
			data[clbl[i]][1][cseq[i]]
			])
		(opt, err) = net.drive_model(v, [])
		plbl[i] = np.argmax(opt)
	crt = 0
	cnt = np.zeros(len(r), 'int')
	for i in xrange(cnum):
		crt += bool(clbl[i]==plbl[i])
		for t in xrange(len(r)):
			cnt[t] += bool(plbl[i]==t)
	print 1.0*crt/cnum
	#print "couter: ", cnt
	# draw preditced points
	pdata = range(len(r))
	for i in xrange(len(r)):
		pdata[i] = np.zeros([2, cnt[i]])
	cnt = np.zeros(len(r), 'int')
	for i in xrange(cnum):
		pdata[plbl[i]][0][cnt[plbl[i]]] = data[clbl[i]][0][cseq[i]]
		pdata[plbl[i]][1][cnt[plbl[i]]] = data[clbl[i]][1][cseq[i]]
		cnt[plbl[i]] += 1
# execute the program
main()

# END OF FILE
