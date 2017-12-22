# -*- coding:utf8 -*-
# test for creating my own network
# using tensorflow

import tensorflow as tf
import numpy as np
import os
import sys

import tf_hlconvnet as hl

label_list = [
        '中国-051B-051C型-驱逐舰',
        '中国-051型-驱逐舰',
        '中国-052型-驱逐舰',
        '中国-053H-护卫舰',
        '中国-053H1-护卫舰',
        '中国-053H1G型-护卫舰',
        '中国-053H1Q-护卫舰',
        '中国-053H2-护卫舰',
        '中国-053H2G-护卫舰',
        '中国-053H3-护卫舰',
        '中国-053K-护卫舰',
        '中国-054-护卫舰',
        '中国-054A-护卫舰',
        '中国-056-护卫舰',
        '中国-071型大型船坞登陆舰-登陆舰',
        '中国-65型-护卫舰',
        '中国-6601型-护卫舰',
        '中国-6607型鞍山级-驱逐舰',
        '中国-中型登陆舰-登陆舰',
        '中国-反潜舰',
        '中国-大型登陆舰-登陆舰',
        '中国-导弹艇',
        '中国-小型登陆舰-登陆舰',
        '中国-扫雷舰',
        '中国-气垫船-登陆舰',
        '中国-现代级-驱逐舰',
        '中国-补给舰',
        '中国-鱼雷艇'
]

def main(args):
    net = hl.hlconvnet(N=1) # N is the batch
    net.build_graph()
    net.restore('/home/hiroki/git/hln/tf_notes/MODEL_20171217_192011/')
    pred, prob = net.predict('/home/hiroki/ships/raw_data/test_one/')
    print 'prediction:'
    print label_list[pred[0][0]]
    print 'with probablity:'
    print prob[0][0]

tf.app.run()

