# Traditional SOM model

import numpy as np
import matplotlib.pyplot as pl

class SOM:
	def __init__(self):
		print 'Creating SOM model instance!'
	def build_model(self, dim):
		self.dim = dim
		self.eta = 0.3
		self.rad = 3
		self.dec = 0.2
		self.ilayer = 