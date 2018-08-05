import tensorflow as tf
import tf_util as tf_util
import sys
sys.path.append('./layers')
from pooling import PoolingLayer
from convlution_layer import ConvLayer
from convlayer_elements import ConvElements


class Network:
    def __init__(self,conf):
        self.conf = conf

    def build_network(self,pointclouds_pl,is_training,is_eveluate,bn_decay = None):
        with_bn = self.conf.get_bool('with_bn')
        batch_size = pointclouds_pl.get_shape()[0].value
        num_point = pointclouds_pl.get_shape()[1].value
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

