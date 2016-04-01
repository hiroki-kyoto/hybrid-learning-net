# procedure for starting the application of instance of HLNN
import numpy as np
import matplotlib.pyplot as pl
from hlnn import HLNN

def main():
	print 'HLNN Instance Running Test'
	net = HLNN()
	###############################################
	# generate samples of test
	###############################################
	number = 500
	r = np.array([10, 10, 10])
	c = np.array([
		[16, 45],
		[18, 27],
		[33, 38]
		])
	# plot them out
	data = range(3)
	for i in xrange(0, len(r)):
		rou = np.random.rand(number)*r[i]
		theta = np.random.rand(number)*2*np.pi
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
	pl.legend('Classification of Random Topologies')
	#pl.show()
	pl.draw()
	fig1.savefig("test1")
	##################################################
	# applying model to test samples
	##################################################
	net.set_net_dim([2, 50, len(r)]);
	net.build_model()
	# prepare unsupervised and supervised data
	unum = 3000 # unsupervised learning
	snum = 500  # supervised learning
	ulbl = np.random.randint(len(r), size=unum)
	useq = np.random.randint(number, size=unum)
	for i in xrange(unum):
		v = np.array([
			data[ulbl[i]][0][useq[i]],
			data[ulbl[i]][1][useq[i]]
			])
		net.drive_model(v, []) # feedback=[] means no feedback
	print "Model unsupervised learning process done!"
	# supervised learning
	slbl = np.random.randint(len(r), size=snum)
	sseq = np.random.randint(number, size=snum)
	serr = np.zeros(snum)
	for i in xrange(snum):
		v = np.array([
			data[slbl[i]][0][sseq[i]],
			data[slbl[i]][1][sseq[i]]
			])
		f = np.zeros(len(r)) + 0.25
		f[slbl[i]] = 0.75
		serr[i] = net.drive_model(v, f)
	# plot error figure
	fig2 = pl.figure()
	pl.plot(xrange(snum), serr)
	pl.show()
	pl.draw()
	fig2.savefig("error")
	# check correctness
	cnum = 100
	clbl = np.random.randint(len(r), size=cnum)
	cseq = np.random.randint(number, size=cnum)
	crt = 0
	for i in xrange(cnum):
		v = np.array([
			data[clbl[i]][0][cseq[i]],
			data[clbl[i]][1][cseq[i]]
			])
		f = clbl[i]
		p = np.argmax(net.drive_model(v, []))
		if p==f:
			crt += 1
	print "correctness: ", 1.0*crt/cnum 



if __name__ == '__main__':
	main()