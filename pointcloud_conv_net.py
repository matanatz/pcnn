import tensorflow as tf
import tf_util as tf_util
import sys
sys.path.append('./layers')
from pooling import PoolingLayer
from convolution_layer import ConvLayer
from convlayer_elements import ConvElements


class Network:
    def __init__(self,conf):
        self.conf = conf

    def build_network(self,pointclouds_pl,is_training,is_eveluate,bn_decay = None):
        with_bn = self.conf.get_bool('with_bn')
        batch_size = pointclouds_pl.get_shape()[0].value
        num_point = pointclouds_pl.get_shape()[1].value

        if (self.conf['with_rotations']):
            cov = self.tf_cov(pointclouds_pl)
            _, axis = tf.self_adjoint_eig(cov)
            axis = tf.where(tf.linalg.det(axis) < 0, tf.matmul(axis, tf.tile(
                tf.constant([[[0, 1], [1, 0]]], dtype=tf.float32), multiples=[axis.get_shape()[0], 1, 1])), axis)

            indicies = [[[b, 0, 0], [b, 2, 0], [b, 0, 2], [b, 2, 2]] for b in list(range(batch_size))]
            updates = tf.reshape(axis, [batch_size, -1])
            updates = tf.reshape(tf.matmul(
                tf.tile(tf.constant([[[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]], dtype=tf.float32),
                        multiples=[batch_size, 1, 1]), tf.expand_dims(updates, axis=-1)), shape=[batch_size, -1])

            alignment_transform = tf.scatter_nd(indices=indicies, updates=updates,
                                                shape=[batch_size, 3, 3]) + tf.expand_dims(
                tf.diag([0.0, 1.0, 0.0]), axis=0)
            mean_points = tf.reduce_mean(pointclouds_pl, axis=1, keepdims=True)
            pointclouds_pl = tf.matmul(pointclouds_pl - mean_points, alignment_transform) + mean_points

        ps_function_pl = tf.concat([pointclouds_pl,tf.ones(shape=[batch_size,num_point,1],dtype=tf.float32)],axis=2)

        pool_sizes_sigma = self.conf.get_list('pool_sizes_sigma')
        spacing = self.conf.get_float('kernel_spacing')

        network = ps_function_pl
        input_channel = network.get_shape()[2].value

        blocks = self.conf.get_list('blocks_out_channels')
        for block_index,block in enumerate(blocks):
            block_elm = ConvElements(pointclouds_pl, 1. * tf.reciprocal(tf.sqrt(tf.cast(pointclouds_pl.get_shape()[1].value,tf.float32))),spacing,self.conf.get_float('kernel_sigma_factor'))
            for out_index,out_channel in enumerate(block):
                network = ConvLayer(input_channel, block_elm, out_channel, '{0}_block_{1}'.format(block_index,out_index),is_training).get_layer(network,with_bn,bn_decay,self.conf.get_bool('interpolation'))
                input_channel = out_channel
            pointclouds_pl, network = PoolingLayer(block_elm, out_channel, out_channel,
                                               int(pool_sizes_sigma[block_index + 1][0])).get_layer(network,is_subsampling=self.conf.get_bool('subsampling'),use_fps= tf.logical_or(is_training,is_eveluate))

        network = tf.reshape(network, [batch_size, -1])
        network = tf_util.fully_connected(network, self.conf.get_int('fc1.size'), bn=True, is_training=is_training,
                                      scope='fc1', bn_decay=bn_decay)
        network = tf_util.dropout(network, keep_prob=self.conf.get_float('dropout.keep_prob'), is_training=is_training,
                                  scope='dp1')
        network = tf_util.fully_connected(network, self.conf.get_int('fc2.size'), bn=True, is_training=is_training,
                                      scope='fc2', bn_decay=bn_decay)
        network = tf_util.dropout(network, keep_prob=self.conf.get_float('dropout.keep_prob'), is_training=is_training,
                                  scope='dp2')
        network = tf_util.fully_connected(network, 40, activation_fn=None, scope='fc3')

        return network

    def get_loss(self, pred, label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
        return tf.reduce_mean(loss)

    def tf_cov(self, x):
        x = tf.transpose(tf.gather(tf.transpose(x, [2, 1, 0]), [0, 2]), [2, 1, 0])
        mean_x = tf.reduce_sum(x, axis=1, keepdims=True)
        mx = tf.matmul(tf.transpose(mean_x, [0, 2, 1]), mean_x)
        vx = tf.einsum('bij,bik->bjk', x, x)
        num = tf.cast(tf.shape(x)[1], tf.float32)
        cov_xx = 1. / num * (vx - (1. / num) * mx)
        return cov_xx