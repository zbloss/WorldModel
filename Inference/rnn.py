# Building the MDN-RNN model
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LayerNormBasicLSTMCell
from tensorflow.nn.rnn_cell import DropoutWrapper
from tensorflow.nn import dynamic_rnn


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

        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture
        INWIDTH = hps.input_seq_width
        OUTWIDTH = hps.output_seq_width
        LENGTH = self.hps.max_seq_len

        if hps.is_training == True:
            self.global_step = tf.Variable(
                initial_value=0, name='global_step', trainable=False)

        cell_fn = LayerNormBasicLSTMCell

        # Setting recurrent, input, and output dropouts
        if self.hps.use_recurrent_dropout == 0:
            use_recurrent_dropout = False
        else:
            use_recurrent_dropout = True

        if self.hps.use_input_dropout == 0:
            use_input_dropout = False
        else:
            use_input_dropout = True

        if self.hps.use_output_dropout == 0:
            use_output_dropout = False
        else:
            use_output_dropout = True

        if self.hps.use_layer_norm == 0:
            use_layer_norm = False
        else:
            use_layer_norm = True

        if use_recurrent_dropout == True:
            cell = cell_fn(num_units=hps.rnn_size, layer_norm=use_layer_norm,
                           dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(num_units=hps.rnn_size, layer_norm=use_layer_norm)

        if use_input_dropout == True:
            cell = DropoutWrapper(
                cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout == True:
            cell = DropoutWrapper(
                cell, output_keep_prob=self.hps.output_dropout_prob)

        self.cell = cell
        self.sequence_length = LENGTH
        self.input_x = tf.placeholder(dtype=tf.float32, shape=[
                                      self.hps.batch_size, LENGTH, INWIDTH])
        self.output_x = tf.placeholder(dtype=tf.float32, shape=[
                                       self.hps.batch_size, LENGTH, OUTWIDTH])
        actual_input_x = self.input_x
        self.initial_state = cell.zero_state(
            batch_size=hps.batch_size, dtype=tf.float32)
        NOUT = OUTWIDTH * KMIX * 3

        with tf.variable_scope('RNN'):
            # weights
            output_w = tf.get_variable(name="output_w", shape=[
                                       self.hps.rnn_size, NOUT])
            # biases
            output_b = tf.get_variable(name="output_b", shape=[NOUT])

        # deterministic output of the RNN
        output, last_state = dynamic_rnn(
            cell=cell,
            inputs=actual_input_x,
            initial_state=self.initial_state,
            dtype=tf.float32,
            swap_memory=True,
            scope='RNN'
        )

    ### Build the MDN ###
    output = tf.reshape(output, shape=[-1, hps.rnn_size])

    # this is where we gather the deterministic output of the RNN so that we can
    # feed it into the MDN. Connecting the Networks!

    # This is the hidden layer that bridges the gap between the output of the RNN
    # and the input of the MDN 
    output = tf.nn.xw_plus_b(x=output, weights=output_w, biases=output_b)
    output = tf.reshape(output, shape=[-1, KMIX*3])



