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
	number = 100
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
	net.set_net_dim([2, 10, 1]);
	net.set_scale(100, 3)
	net.build_model()
	# prepare unsupervised and supervised data
	unum = 500 # unsupervised learning
	snum = 200  # supervised learning
	ulbl = np.random.randint(len(r), size=unum)
	useq = np.random.randint(number, size=unum)
	uerr = np.zeros(unum)
	for i in xrange(unum):
		v = np.array([
			data[ulbl[i]][0][useq[i]],
			data[ulbl[i]][1][useq[i]]
			])
		[uopt, uerr[i]] = net.drive_model(v, []) # feedback=[] means no feedback
	# plot unsupervised error level
	uerrfig = pl.figure()
	pl.plot(xrange(unum), uerr, "-")
	pl.xlabel("Iteration Time")
	pl.ylabel("Unsupervised Error")
	pl.legend("Unsupervised Learning Error Curve")
	#pl.show()
	pl.draw()
	uerrfig.savefig("uerrfig")
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
		serr[i] = net.drive_model(v, np.array([slbl[i]]))
	# plot error figure
	fig2 = pl.figure()
	pl.plot(xrange(snum), serr, "-")
	#pl.show()
	pl.draw()
	fig2.savefig("error")
	# check correctness
	cnum = 100
	clbl = np.random.randint(len(r), size=cnum)
	cseq = np.random.randint(number, size=cnum)
	plbl = np.random.randint(len(r), size=cnum)
	for i in xrange(cnum):
		v = np.array([
			data[clbl[i]][0][cseq[i]],
			data[clbl[i]][1][cseq[i]]
			])
		[opt, err] = net.drive_model(v, [])
		print opt
		plbl[i] = np.argmin(np.abs(np.arange(3)-opt))
	crt = 0
	cnt = np.zeros(len(r), 'int')
	for i in xrange(cnum):
		crt += bool(clbl[i]==plbl[i])
		for t in xrange(len(r)):
			cnt[t] += bool(plbl[i]==t)
	print "correctness: ", 1.0*crt/cnum
	print "couter: ", cnt
	# draw preditced points
	pdata = range(len(r))
	for i in xrange(len(r)):
		pdata[i] = np.zeros([2, cnt[i]])
	cnt = np.zeros(len(r), 'int')
	for i in xrange(cnum):
		pdata[plbl[i]][0][cnt[plbl[i]]] = data[clbl[i]][0][cseq[i]]
		pdata[plbl[i]][1][cnt[plbl[i]]] = data[clbl[i]][1][cseq[i]]
		cnt[plbl[i]] += 1
	pfig = pl.figure()
	pl.plot(pdata[0][0], pdata[0][1], "*")
	pl.plot(pdata[1][0], pdata[1][1], ".")
	pl.plot(pdata[2][0], pdata[2][1], "x")
	pl.legend("Prediction Result")
	pl.xlabel("x")
	pl.ylabel("y")
	pl.draw()
	pfig.savefig("prediction")


if __name__ == '__main__':
	main()

# END OF FILE
