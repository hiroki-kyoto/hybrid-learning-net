# hybrid learning kernel for embedding in
# arbitrary neural layers

import numpy as np
import matplotlib.pyplot as pl


# signal normalization
def sig_norm(x):
    _max = x.max()
    _min = x.min()
    _scale = _min/(_max-_min+1e-6)
    return (_max*_scale)/(x+1e-6)-_scale

# embedding:
# @x : the input feature
# @w : weight matrix for the self organizing map
#      dimension with ([NEXT_LAYER_DIM, DIM_OF(X)])
def hybrid_learn_embed(x, w):
    return sig_norm(np.linalg.norm(w-x, axis=1))

