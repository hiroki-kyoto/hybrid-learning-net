# Building a operator for constructing 
# hybrid learning layer
# Please refer to : 
# https://github.com/hiroki-kyoto/hybrid-learning-net/blob/master/ecrc-procs/hybrid_learning_net.pdf
# The hybrid learning embedding can be applied
# in any formation of layer, and brings no
# more supervised parameter!
# which means, it will not effect the
# complexity of supervised training,
# otherwise it learns the presentation of
# activation when a neural layer is
# exposed in a specific input feature space.

# The implementation for tensorflow operator : HLCONV

