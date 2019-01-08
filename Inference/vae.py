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

            # now we have the data into a one-dimensional array.
            # it is now ready to be pushed into our Variational Auto-Encoder
            h = tf.reshape(tensor=h, shape=[-1, 2*2*256])

            # mean layer of the VAE
            self.mu = tf.layers.dense(inputs=h, units=self.z_size, name='encoder_fc_mu')
            
            # standard deviation layer of the VAE
            self.logvar = tf.layers.dense(inputs=h, units=self.z_size, name='encoder_fc_logvar')
            self.sigma = tf.exp(self.logvar / 2.0)

            self.epsilon = tf.random_normal([self.batch_size, self.z_size])

            # final latent vector
            self.z = self.mu + self.sigma * self.epsilon

            # building the decoder
            h = tf.layers.dense(inputs=self.z, units=self.z_size, name='decoder_fc')

            h = tf.reshape(tensor=h, shape=[-1, 1, 1, 2*2*256])

            # first inverted Conv layer
            h = tf.layers.conv2d(inputs=h, filters = 128, kernel_size=5, strides=2, activation=tf.nn.relu, name='decoder_deconv1')
            
            # second inverted Conv layer
            h = tf.layers.conv2d(inputs=h, filters = 64, kernel_size=5, strides=2, activation=tf.nn.relu, name='decoder_deconv2')
            
            # third inverted Conv layer
            h = tf.layers.conv2d(inputs=h, filters = 32, kernel_size=6, strides=2, activation=tf.nn.relu, name='decoder_deconv3')
            
            # fourth & final inverted Conv layer
            self.y = tf.layers.conv2d(inputs=h, filters = 3, kernel_size=6, strides=2, activation=tf.nn.sigmoid, name='decoder_deconv4')

            # implement the training operations
            # we want to train the network such that the processed image y, matches the original image x
            if self.is_training == True:
                
                self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
                
                # calculating mse loss
                self.r_loss = tf.reduce_sum(tf.square(self.x - self.y), reduction_indices=[1, 2, 3])
                self.r_loss = tf.reduce_mean(self.r_loss)

                # calculating the Kullback–Leibler loss
                self.kl_loss = -0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices=1)

                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)

                self.kl_loss = tf.reduce_mean(self.kl_loss)

