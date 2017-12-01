# visualize the theory for
# input sensitive connection

import numpy as np
import matplotlib.pyplot as pl

import HLKernel as hl

x = np.array([[0.05,0.3,0.1]])
w = np.random.rand(1000,3)
y = hl.hl_embed(x, w)

pl.plot(y)
pl.show()
pl.plot(np.sort(y))
pl.show()

_flag = False

for _i in xrange(800):
    w = hl.train_embed(x, w, y)
    y = hl.hl_embed(x, w)
    if _i==100:
        _flag = True
    elif _i==200:
        _flag = True
    elif _i==400:
        _flag = True
    elif _i==800:
        _flag = True
    else:
        _flag = False
    if _flag:
        pl.plot(y)
        pl.show()
        pl.plot(np.sort(y))
        pl.show()

