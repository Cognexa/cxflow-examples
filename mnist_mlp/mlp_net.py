import logging

import cxflow_tensorflow as cxtf
import tensorflow as tf
import tensorflow.contrib.keras as K


class MNIST_MLP(cxtf.BaseModel):
    """Simple 2-layered MLP for MNIST hand-written digits recognition."""

    def _create_model(self, hidden: int=100) -> None:
        logging.debug('Constructing placeholders')
        images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='images')
        labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')

        logging.debug('Constructing MLP')
        with tf.variable_scope('dense1'):
            hidden_activations = K.layers.Dense(hidden, activation=tf.nn.relu)(tf.contrib.layers.flatten(images))
        with tf.variable_scope('dense2'):
            logits = K.layers.Dense(10)(hidden_activations)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(logits, 1, name='predictions')
        tf.equal(predictions, labels, name='accuracy')
