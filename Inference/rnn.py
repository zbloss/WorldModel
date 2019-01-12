# Building the MDN-RNN model
import numpy as np
import tensorflow as tf


class MDNRNN(object):

    # initializing all params and variables of the MDNRNN class
    def __init__(self, hps, reuse=False, gpu_mode=False):
        
        self.hps = hps

        with tf.variable_scope('mdn_rnn', reuse=reuse):
            if gpu_mode == False:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model is using the CPU...')
                    self.g = tf.Graph()
                    with self.g.as_default():
                        self.build_model(hps)
            else:
                tf.logging.info('Model is using the GPU...')
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model(hps)

        self._init_session()

    # creating the VAE model architecture

    def build_model(self, hps):