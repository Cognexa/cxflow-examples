import logging

import cxflow_tensorflow as cxtf
import tensorflow as tf
import tensorflow.contrib.keras as K


class MajorityNet(cxtf.BaseModel):
    """Simple 2-layered MLP for majority task."""

    def _create_model(self, hidden, dim):
        logging.debug('Constructing placeholders matching the model.inputs')
        x = tf.placeholder(dtype=tf.float32, shape=[None, dim], name='x')
        y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        logging.debug('Constructing MLP model')
        net = K.layers.Dense(hidden)(x)
        y_hat = K.layers.Dense(1)(net)[:, 0]

        logging.debug('Constructing loss and outputs matching the model.outputs')
        tf.pow(y - y_hat, 2, name='loss')
        predictions = tf.greater_equal(y_hat, 0.5, name='predictions')
        tf.equal(predictions, tf.cast(y, tf.bool), name='accuracy')
