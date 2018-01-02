# test for creating my own network
# using tensorflow

import tensorflow as tf
import tf_hlconvnet as hl

import setting as st

def main(argv):
    net = hl.hlconvnet(N=st.__BATCH_SIZE__)
    conf_file = open(st.__JSON_CONF_FILE__, 'rt')
    conf_json = conf_file.read()
    net.build_graph(conf_json)
    net.offline_supervised(
            st.__DATA_PATH__ % st.__TRAIN_DIR__
            )
    net.offline_unsupervised(
            st.__DATA_PATH__ % st.__UNLABELED_DIR__
            )
    net.save_log_at(st.__LOG_PATH__)
    net.save_model_at(st.__MODEL_PATH__)
    net.train(st.__EPOC__, st.__UNS_PER__, st.__SUP_PER__)

tf.app.run()

