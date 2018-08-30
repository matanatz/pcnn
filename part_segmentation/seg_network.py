
import tensorflow as tf
import numpy as np

import sys
sys.path.append('../layers')
from convlayer_elements import ConvElements
import tf_util
from pooling import PoolingLayer
from convolution_layer import ConvLayer
from deconvolution_layer import DeconvLayer
from pyhocon import ConfigFactory

class SegNetwork:
    def __init__(self, conf):
        self.conf = conf

    def get_network_model(self, pointclouds_pl,input_label, cat_num, part_num, \
		batch_size, num_point, weight_decay,is_training, bn_decay=None):
        with_bn = self.conf.get_bool('with_bn')
        batch_size = pointclouds_pl.get_shape()[0].value
        num_point = pointclouds_pl.get_shape()[1].value
        ps_function_pl = tf.concat([pointclouds_pl, tf.ones(shape=[batch_size, num_point, 1], dtype=tf.float32)],
                                   axis=2)

        pool_sizes_sigma = self.conf.get_list('pool_sizes_sigma')  # [256,64,4]
        spacing = np.float32(self.conf.get_float('kernel_spacing'))

        # first scale
        network = ps_function_pl

        input_channel = 4

        blocks = self.conf.get_list('blocks_out_channels')
        pointclouds_pl_down = []

        for block_index, block in enumerate(blocks):
            block_elm = ConvElements(pointclouds_pl, np.float32(1.)/np.sqrt(pointclouds_pl.get_shape()[1].value,dtype=np.float32), spacing,
                                          np.float32(self.conf.get_float('kernel_sigma_factor')))
            for out_index, out_channel in enumerate(block):
                network = ConvLayer(input_channel, block_elm, out_channel,
                                     '{0}_block_{1}'.format(block_index, out_index), is_training).get_layer(
                    network, with_bn, bn_decay,False,True)
                input_channel = out_channel

            pointclouds_pl_down.append([pointclouds_pl, network])
            pointclouds_pl, network = PoolingLayer(block_elm, out_channel, out_channel,
                                                   int(pool_sizes_sigma[block_index + 1][0])).get_layer(network,use_fps = tf.constant(False),is_subsampling = self.conf.get_bool("is_subsampling"))

        pointclouds_pl_down.append([pointclouds_pl,None])


        one_hot_label_expand = tf.tile(tf.expand_dims(input_label, axis=1),[1,pointclouds_pl_down[-1][0].get_shape()[1].value,1])
        network = tf.concat(axis=2, values=[network, one_hot_label_expand])

        for i in range(len(pointclouds_pl_down) -1 ,0,-1):

            deconvnet = DeconvLayer(pointclouds_pl_down[i][0],network,pointclouds_pl_down[i-1][0]).get_layer()

            block_elm = ConvElements(pointclouds_pl_down[i - 1][0],
                              tf.reciprocal(tf.sqrt(1.0 * pointclouds_pl_down[i - 1][0].get_shape()[1].value)), spacing,
                              self.conf.get_float('kernel_sigma_factor'))

            convnet = ConvLayer(deconvnet.get_shape()[-1].value, block_elm, pointclouds_pl_down[i-1][1].get_shape()[-1].value, "{0}_deconv".format(i),
                           is_training).get_layer(deconvnet, with_bn , bn_decay, False,True)

            network = tf.concat([pointclouds_pl_down[i-1][1],convnet],axis=2)


            if (self.conf.get_bool('is_dropout') and i  <= 2):
                network = tf_util.dropout(network,is_training,"dropout_{0}".format(i),keep_prob = self.conf.get_float('dropout.keep_prob'))

        block_elm = ConvElements(pointclouds_pl_down[0][0],tf.reciprocal(tf.sqrt(1.0 * pointclouds_pl_down[0][0].get_shape()[1].value)),
                                      spacing,
                                      self.conf.get_float('kernel_sigma_factor')) \

        network = ConvLayer(network.get_shape()[-1].value, block_elm,
                                 part_num, "{0}_deconv".format('last'),
                           is_training).get_layer(network, False, bn_decay,False ,False)

        return network

    def get_loss(self, seg_pred, seg):

        # size of seg_pred is batch_size x point_num x part_cat_num
        # size of seg is batch_size x point_num
        per_instance_seg_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
        seg_loss = tf.reduce_mean(per_instance_seg_loss)

        per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

        total_loss =  seg_loss

        return total_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res