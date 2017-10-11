# graph maker
import numpy as np
import matplotlib.pyplot as pl
import sys
import os

# compare the supervised error
x = np.load('serr-bpnn-' + sys.argv[1] + '.npy')
y = np.load('serr-hlnn-' + sys.argv[1] + '.npy')

# draw in same graph for comparison
pl.figure(figsize=(8,2))
pl.plot(x[0:50], 'y+-')
pl.plot(y[0:50], 'ro-')
pl.legend(['FCN', 'HLN'])
pl.ylabel('Training Loss')
pl.tight_layout()
pl.savefig('serr-' + sys.argv[1] + '.pdf', dpi=100)

del x
del y

# compare test accuracy
x = np.load('tacc-bpnn-' + sys.argv[1] + '.npy')
y = np.load('tacc-hlnn-' + sys.argv[1] + '.npy')

# draw in same graph for comparison
pl.figure(figsize=(8,2))
pl.plot(x[0:50], 'y+-')
pl.plot(y[0:50], 'ro-')
pl.legend(['FCN', 'HLN'])
pl.ylabel('Test Accuracy')
pl.tight_layout()
pl.savefig('tacc-' + sys.argv[1] + '.pdf', dpi=100)

del x
del y

# compare test accuracy
x = np.load('sspa-bpnn-' + sys.argv[1] + '.npy')
y = np.load('sspa-hlnn-' + sys.argv[1] + '.npy')

# draw in same graph for comparison
pl.figure(figsize=(8,2))
pl.plot(x[0:50]*100, 'y+-')
pl.plot(y[0:50]*100, 'ro-')
pl.legend(['FCN', 'HLN'])
pl.ylabel('Activation Ratio(%)')
pl.tight_layout()
pl.savefig('sspa-' + sys.argv[1] + '.pdf', dpi=100)

del x
del y



