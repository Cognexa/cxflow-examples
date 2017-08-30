import cxflow_tensorflow as cxtf
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers import Embedding, GRU, Dropout, Dense, Bidirectional


class SimpleGRU(cxtf.BaseModel):
    """Simple 2-layered MLP for majority task."""

    def _create_model(self, dropout: int=0.5, embedding_dim: int=256, gru_dim: int=128):
        x = tf.placeholder(dtype=tf.float32, shape=[None, self._dataset.maxlen], name='x')
        y = tf.placeholder(dtype=tf.int64, shape=[None], name='y')

        with tf.variable_scope('emb1'):
            net = Embedding(self._dataset.vocab_size, embedding_dim)(x)
            net = Dropout(0.1)(net, training=self.is_training)
        with tf.variable_scope('gru2'):
            net = Bidirectional(GRU(gru_dim, return_sequences=True))(net)
            net = tf.reduce_mean(net, axis=1)
        with tf.variable_scope('dense4'):
            net = Dropout(dropout)(net,  training=self.is_training)
            logits = Dense(2, activation=None)(net)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(logits, 1, name='predictions')
        tf.equal(predictions, y, name='accuracy')
