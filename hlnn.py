# HLNN.py
# class of Hybrid Learning Neural Network

class HLNN :
	"
	som_dim: dimensionality of SOM
	som_eta: learning rate of SOM
	som_rad: neighborhood radius
	som_dec: decrease rate of correctness
	bp_layer: layer number of BPNN
	bp_dim: dimensionalities of all BPNN layers
	bp_eta: BPNN learning reate
	bp_coe: BPNN learning rate coefficient as cur*coe+last*(1-coe)
	"

	def __init__(self, \
	som_dim=10, \
	som_eta=0.2, \
	som_rad=3, \
	som_dec=0.5, \
	bp_layer=1, \
	bp_dim=[1], \
	bp_eta=0.6, \
	bp_coe=0.5) :
		print som_dim

