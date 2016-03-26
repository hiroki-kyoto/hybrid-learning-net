# procedure for starting the application of instance of HLNN
from hlnn import HLNN

def main():
	print 'HLNN Instance Running Test'
	net = HLNN()
	print(net.ready)
	net.set_net_dim([2,100,2])
	print(net.ready)

main()	
