# hybrid learning kernel for embedding in
# arbitrary neural layers

import numpy as np

# define some constants for best
# parameter tunning
__global_learn_rate = 0.01

# signal normalization
def sig_norm(x):
    _max = x.max()
    _min = x.min()
    _scale = _min/(_max-_min+1e-6)
    return (_max*_scale)/(x+1e-6)-_scale

# embedding:
# @@ : y = f(x, w)
# @x : the input feature vector
# @w : weight matrix for the self organizing map
#      dimension with ([NEXT_LAYER_DIM, DIM_OF(X)])
# @y : the normalized signal
def hl_embed(x, w):
    return sig_norm(np.linalg.norm(w-x, axis=1))

# hybrid learning kernel training

# unsupervised training for hybrid
# learning embeddings
# @@ : w' = f(x, w, y) [w' is updated]
# @x : the input feature vector
# @w : weight matrix for the self organizing map
# @y : the normalized signal
# @w': the updated weight matrix referred from w
def train_embed(x, w, y):
    if len(x.shape) == 1:
        _n = x.shape[0]
    else:
        _n = x.shape[1]
    for _i in xrange(_n):
        w[:,_i] += __global_learn_rate*y*((x - w)[:,_i])

    return w

