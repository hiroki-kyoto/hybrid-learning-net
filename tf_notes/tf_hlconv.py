# hybrid-learning embedding for convolutional
# operators implemented by tensorflow (1.0+)

import numpy as np
import tensorflow as tf

# signal polarization
# @x : A 4D tensor with shape [batch, height, width, depth]
# ret: Returning a 4D tensor with shape remained
#      polarized on the first dim( axis=3)
def sig_polar(x):
    _max = tf.reduce_max(x, axis=0)
    _min = tf.reduce_min(x, axis=0)
    _scale = _min/(_max-_min+1e-6)
    return (_max*_scale)/(x+1e-6)-_scale


# Make parameter from configuration
# Calling with keyword arguments is also supported
# @n      : total number of kernels
# @depth  : depth or channel of kernel
# @ksizes : must be [1, height, width, 1]
# @init   : initialization manner, random or zero
# @name   : parameter tensor name, retrived by
#           tf.get_default_graph().get_tensor_by_name([str])
# @ret    : return two named parameters: (uparam, sparam)
#           uparam : unsupervised parameter
#           sparam : supervised parameter
#           they both have the same shape with [kernel, depth]
def hlconv_make_params(
        n=None,
        depth=None,
        ksizes=None, 
        init='random', 
        name='hlconv',
        **kwargs
    ):
    # named args
    if 'n' in kwargs:
        n = kwargs['n']
    if 'depth' in kwargs:
        depth = kwargs['depth']
    if 'ksizes' in kwargs:
        ksizes = kwargs['ksizes']
    if 'init' in kwargs:
        init = kwargs['init']
    if 'name' in kwargs:
        name = kwargs['name']
    # simple validation for args
    assert ksizes[0]==1
    assert ksizes[3]==1
    
    height = ksizes[1]
    width = ksizes[2]
    _size = height*width*depth

    if init == 'random':
        return (
                tf.Variable(
                    tf.random_uniform(
                        [n,_size]
                    ),
                    trainable=False, # stops gradient
                    name=name+'-u'
                ),
                tf.Variable(
                    tf.random_uniform(
                        [n,_size]
                    ),
                    trainable=True,
                    name=name+'-s'
                )
        )
    elif init == 'zero':
        return (
                tf.Variable(
                    tf.zeros(
                        [n,_size]
                    ),
                    trainable=False,
                    name=name+'-u'
                ),
                tf.Variable(
                    tf.zeros(
                        [n,_size]
                    ),
                    trainable=True,
                    name=name+'-s'
                )
        )
    else:
        raise NameError('unknown init manner:' + str(init))

# @x      : must be a tensorflow input 4D tensor
#           [batch_size, height, width, channel]
# @uw     : must be a 2D tensor parameter with dimension
#           [n, kheight*kwidth*kdepth], plz use the
#           generating method: [hlconv_make_params]
#           unsupervised weights of hlconv operator
# @sw     : must be a 2D tensor parameter with dimension
#           [n, kheight*kwidth*kdepth], plz use the
#           generating method: [hlconv_make_params]
#           supervised weights of hlconv operator
# @ksizes : kernel sizes [height, width, depth] must be
#           the same as the ones used to generate [w]
# @strides: defined as in convolution operator: patch offset
# @padding: defined as in convolution operator: 'VALID' or 'SAME'
# @name   : tensor name, got by 
#           tf.get_default_graph().get_tensor_by_name([str])
# @ret    : return a tuple of (run_op, unsupervised_train_op)
#           Notice that this operator [hlconv] will not include
#           the unsupervised parameter [uw] into the final optimizer,
#           instead, it is required to implement the training 
#           operator standalone.
def hlconv(
        x=None, 
        uw=None,
        sw=None,
        ksizes=None, 
        strides=None, 
        padding=None, 
        name=None,
        **kwargs
    ):
    # replace default args with kwargs
    if 'x' in kwargs:
        x = kwargs['x']
    if 'uw' in kwargs:
        uw = kwargs['uw']
    if 'sw' in kwargs:
        sw = kwargs['sw']
    if 'ksizes' in kwargs:
        ksizes = kwargs['ksizes']
    if 'strides' in kwargs:
        strides = kwargs['strides']
    if 'padding' in kwargs:
        padding = kwargs['padding']
    if 'name' in kwargs:
        name = kwargs['name']
    # simple validation
    assert len(ksizes)==4
    assert len(x.shape.as_list())==4
    assert ksizes[0]==1

    kh = ksizes[1]
    kw = ksizes[2]
    depth = x.shape.as_list()[3]
    kn = uw.shape.as_list()[0]
    assert kh*kw*depth==uw.shape.as_list()[1]
    
    patches = tf.extract_image_patches(
            x,
            ksizes = ksizes,
            strides = strides,
            rates = [1,1,1,1],
            padding = padding,
            name = name + '-p'
    )
    # create output tensors
    y = list(range(kn))
    fields = list(range(kn))
    arrivals = list(range(kn))
    # construct sub-operator-graph
    for _i in xrange(kn):
        fields[_i] = tf.norm(uw[_i]-patches, axis=3)
        arrivals[_i] = tf.reduce_sum(sw[_i]*patches, axis=3)
    fields_stacked = tf.stack(fields, axis=0)
    fields_polarized = sig_polar(fields_stacked)
    for _i in xrange(kn):
        y[_i] = fields_polarized[_i]*arrivals[_i]
    return (
            tf.stack(y, axis=3, name=name+'-hlconv'),
            make_unsupervised_train_op(
                uw, 
                patches, 
                fields_polarized
            )
    )

# unsupervised optimizing policy for HLConv operator
__global_learn_rate = 0.01
# @uw               : variable uw
# @patches          : tenosr patches
# @fields_polarized : the polarized fields
# @ret              : return an unsupervised training operator
def make_unsupervised_train_op(
        uw,
        patches, 
        fields_polarized
    ):
    _n = uw.shape.as_list()[0]
    grads = list(range(_n))
    # reshape for dimension broadcasting operators
    dim = fields_polarized.shape.as_list()
    dim.append(1)
    fields_reshaped = tf.reshape(
            fields_polarized,
            dim
    )
    # compute gradients for parameters
    for _i in xrange(_n):
        grads[_i] = fields_reshaped[_i]*(patches-uw[_i])
    grads_stacked = tf.stack(grads, axis=0)
    uw += __global_learn_rate*tf.reduce_sum(grads, axis=(1,2,3))
    return uw # return updated uw as training operator

