# HLConv.py
# A implementation of hybrid learning 
# embedded convolutional operator 
# [with NCHW format input]

# @x: a 3D map, the input vector with dim:
#     [channel, height, width]
# @w: a 3D map, the weight matrix
#     [channel, height, width]

def hl_conv2d(x, w):
    h, w = x.shape[0], x.shape[1]
    sh, sw = w.shape[0], 

def hl_update(
        x, # input
        w, # parameter

