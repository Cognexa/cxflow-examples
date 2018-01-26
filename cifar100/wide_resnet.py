import tensorflow as tf
import cxflow_tensorflow as cxtf
from tensorflow.contrib import slim


def wide_block(inputs, filters, n, layer, dropout, is_training):

    # create first block of residual
    if layer >= 3:
        stride = 2
    else:
        stride = 1

    shortcut = slim.conv2d(inputs, filters, stride=stride)
    net = slim.conv2d(inputs, filters, stride=stride)
    net = slim.batch_norm(net, activation_fn=tf.nn.relu)
    if dropout != 0:
        net = slim.dropout(net, dropout, is_training=is_training)
    net = slim.conv2d(net, filters)
    net += shortcut

    for block in range(int(n)-1):
        shortcut = net
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        net = slim.conv2d(net, filters)
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)
        if dropout != 0:
            net = slim.dropout(net, dropout, is_training=is_training)
        net = slim.conv2d(net, filters)
        net += shortcut
    net = slim.batch_norm(net, activation_fn=tf.nn.relu)

    return net

def wrn_model(inputs, depth, k, weight_decay, dropout, num_classes, is_training=True):

    if ((depth - 4) % 6) != 0:
        raise ValueError('You have to choose depth which is ((depth - 4) % 6) == 0')

    num_filters = [16, 16*k, 32*k, 64*k]
    n = (depth - 4) / 6

    with slim.arg_scope([slim.conv2d],
                        activation_fn=None,
                        stride=1,
                        kernel_size=[3,3],
                        padding='SAME',
                        weights_regularizer=slim.l2_regularizer(weight_decay)):

        net = slim.conv2d(inputs, num_filters[0])
        net = slim.batch_norm(net, activation_fn=tf.nn.relu)

        for layer, filters in enumerate(num_filters[1:]):
            net = wide_block(net, filters, n, layer+2, dropout, is_training)

        net = slim.avg_pool2d(net, 8, stride=1, padding='VALID')
        net = slim.flatten(net)
        net = slim.fully_connected(net, num_classes, activation_fn=None)

    return net

class WideResNet(cxtf.BaseModel):

    def _create_model(self, depth: int, k: int, weight_decay: float=0.0005, dropout: float=0, num_classes: int=100):

        images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        net = wrn_model(images, depth, k, weight_decay, dropout, num_classes, is_training=True)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=net)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(net, 1, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32, name='accuracy'))
