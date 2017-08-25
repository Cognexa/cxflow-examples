import logging

import cxflow_tensorflow as cxtf
import tensorflow as tf
import tensorflow.contrib.keras as K


class MajorityNet(cxtf.BaseModel):
    """Simple 2-layered MLP for majority task."""

    def _create_model(self, hidden):
        logging.debug('Constructing placeholders')
        x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        logging.debug('Constructing MLP')
        with tf.variable_scope('dense1'):
            hidden_activations = K.layers.Dense(hidden)(x)
        with tf.variable_scope('dense2'):
            y_hat = K.layers.Dense(1)(hidden_activations)[:, 0]

        tf.pow(y - y_hat, 2, name='loss')

        logging.debug('Constructing predictions and accuracy')
        predictions = tf.greater_equal(y_hat, 0.5, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(y, tf.bool)), tf.float32, name='accuracy'))
