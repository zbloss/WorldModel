# Building the VAE model
import numpy as np
import tensorflow as tf


class ConvVAE(object):

    # initializing all params and variables of the ConvVAE class
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        self.gpu_mode = gpu_mode

        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if gpu_mode == False:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model is using the CPU...')
                    self._build_graph()
            else:
                tf.logging.info('Model is using the GPU...')
                self._build_graph()

        self._init_session()

    # creating the VAE model architecture

    def _build_graph(self):

        # new self attribute, the graph
        self.g = tf.Graph()
        with self.g.as_default():
            # 64x64x3 - 3 because we are using RGB images
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3])

            # first Conv layer
            h = tf.layers.conv2d(inputs=self.x, filters=32, kernel_size=4,
                                 strides=2, activation=tf.nn.relu, name="encoder_conv1")

            # second Conv layer
            h = tf.layers.conv2d(inputs=h, filters=64, kernel_size=4,
                                 strides=2, activation=tf.nn.relu, name="encoder_conv2")

            # third Conv layer
            h = tf.layers.conv2d(inputs=h, filters=128, kernel_size=4,
                                 strides=2, activation=tf.nn.relu, name="encoder_conv3")

            # fourth Conv layer
            h = tf.layers.conv2d(inputs=h, filters=256, kernel_size=4,
                                 strides=2, activation=tf.nn.relu, name="encoder_conv4")

            h = tf.reshape(tensor=h, shape=[-1, 2*2*256])
