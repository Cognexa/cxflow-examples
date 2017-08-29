import logging

import cxflow_tensorflow as cxtf
import tensorflow as tf
import tensorflow.contrib.keras as K


class SimpleLSTM(cxtf.BaseModel):
    """Simple 2-layered MLP for majority task."""

    def _create_model(self):
        logging.debug('Constructing placeholders')
        x = tf.placeholder(dtype=tf.float32, shape=[None, self._dataset.maxlen], name='x')
        y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')

        with tf.variable_scope('emb1'):
            net = K.layers.Embedding(self._dataset.vocab_size, 256)(x)
        # with tf.variable_scope('lstm2'):
        #     net = K.layers.LSTM(256, return_sequences=True, input_shape=(self._dataset.maxlen, 279))(net)
        # with tf.variable_scope('lstm3'):
        #     net = K.layers.LSTM(256, return_sequences=True, input_shape=(self._dataset.maxlen, 256))(net)
        with tf.variable_scope('lstm4'):
            net = K.layers.LSTM(128, input_shape=(self._dataset.maxlen, 256))(net)
        with tf.variable_scope('dense5'):
            net = K.layers.Dropout(0.5)(net,  training=self.is_training)
            logits = K.layers.Dense(2, activation=None)(net)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(logits, 1, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(predictions, y), tf.float32, name='accuracy'))