# Pattern Connection Network
import numpy as np

# convert maps to patches
# @maps: maps of current layer
# @patches: patches to fill
# @ksize: kernel size, [depth,height,width]
def m2p(maps, patches, ksize):


class pcn(object):
    def __init__(
            self,
            x_dim=None,
            y_dim=None,
            pattern_dims=None,
            learning_rate=0.01,
            decay_rate=0.01
    ):
        print 'init'

        # default settings for pcn
        if x_dim is None:
            x_dim = [256, 256, 3]
        if y_dim is None:
            y_dim = [6]
        if pattern_dims is None:
            pattern_dims = [[3, 3, 32], [3, 3, 16], [3, 3, 8]]

        self.x = np.zeros(x_dim)
        self.y = np.zeros(y_dim)
        self.patterns = [] # variables
        self.maps = [] # maps of pattern
        self.patches = [] # patches of image
        self.conns = [] # variables

        # create patterns and maps of pattern
        _n = len(pattern_dims)
        _sum_of_patterns = 0
        # for each layer
        for i in xrange(_n):
            assert len(pattern_dims[i]) == 3
            # complete the dimension for each pattern
            if i==0:
                pattern_dims[i].insert(0, x_dim[-1])
            else:
                pattern_dims[i].insert(0, pattern_dims[i-1][-1])
            # create pattern variables with random values
            _pattern_len = pattern_dims[i][0]*pattern_dims[i][1]*pattern_dims[i][2]
            _pattern_num = pattern_dims[i][-1]
            # patterns will be normalized to [0,1] using polarization function
            self.patterns.append(np.random.uniform(0,1.0,[_pattern_num, _pattern_len]))
            # only valid convolution is supported, recompute its map width and height
            if i==0:
                _h = x_dim[0] - pattern_dims[i][1] + 1
                _w = x_dim[1] - pattern_dims[i][2] + 1
            else:
                _h = self.maps[-1].shape[0] - pattern_dims[i][1] + 1
                _w = self.maps[-1].shape[1] - pattern_dims[i][2] + 1
            self.maps.append(np.zeros([_h,_w,_pattern_num]))
            # pre-allocation for patches, to avoid possible allocation error
            # during network training or prediction
            self.patches.append(np.zeros(_h,_w,)) ////////
            _sum_of_patterns += _pattern_num

        # activation for every pattern unit(not the map of pattern,
        # but max item of it), namely the max value and its location(x,y)
        # on the map, the location is for network visualized analysis.
        self.act_val = np.zeros([_sum_of_patterns])
        self.act_loc = np.zeros([_sum_of_patterns, 2], np.int32)

        # connections between the patterns and the output units
        # these are also [VARIABLES] but initialized with zero.
        # 0-strength connection means no co-reaction.
        self.conns = np.zeros([y_dim, _sum_of_patterns])

        # back up dimension information
        self.x_dim = x_dim # [H,W,C] # height, width, channel
        self.y_dim = y_dim # [N_DIGITS]
        self.p_dim = pattern_dims # [D,H,W,P] // depth,height,width,pattern
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def data(self, im_dir):
        print 'data'

    def run(self):
        print 'run'

    def save(self):
        print 'save'

    def restore(self, model):
        print 'restore'

    def test(self, im_dir):
        print 'test'

