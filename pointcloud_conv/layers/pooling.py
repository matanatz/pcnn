import tensorflow as tf


class PoolingLayer:
    def __init__(self,block_elm, input_channels, out_channels,k):
        self.out_channels = out_channels
        self.input_channels = input_channels
        self.k = k
        self.block_elm = block_elm

    def fps(self,d,startidx = None):
        batch_size = self.block_elm.batch_size
        num_of_points = self.block_elm.num_of_points

        if (startidx is None):
            idx = tf.random_uniform(shape = [1],minval=0,maxval=num_of_points - 1,dtype=tf.int32)
        else:
            idx = tf.constant(startidx,dtype=tf.int32,shape=[1])

        idx = tf.tile(idx,[batch_size])

        if (self.k == 1):
            idx = tf.expand_dims(idx,axis=1)
        else:

            gather = tf.gather_nd(d, tf.stack([tf.range(batch_size), idx],
                                              axis=1))
            idx = tf.stack([idx, tf.argmax(gather, 1,output_type=tf.int32)], axis=1)

            for step in range(2, max(self.k, 2)):
                mask = tf.reshape(tf.stack([tf.reshape(
                    tf.tile(tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1),
                            [1, idx.get_shape()[-1].value]), shape=[-1]),
                                            tf.reshape(tf.cast(idx, tf.int32), shape=[-1])], axis=1),
                                  shape=[batch_size, -1, 2])

                gather = tf.gather_nd(d, mask)

                idx = tf.concat([idx, tf.expand_dims(tf.argmax(tf.reduce_min(gather, axis=1), 1,output_type=tf.int32), axis=1)], axis=1)

        return idx

    def get_layer(self,network,use_fps,startidx = None,is_subsampling = False):
        batch_size = self.block_elm.batch_size

        distances = self.block_elm.get_distance_matrix()

        idx = tf.cond(use_fps,lambda:self.fps(distances,startidx),lambda:tf.tile(tf.expand_dims(tf.range(self.k),0),[batch_size,1]))
        temp = tf.expand_dims(tf.range(batch_size, dtype=tf.int32), axis=1)
        mask = tf.reshape(tf.stack([tf.reshape(
            tf.tile(temp, [1, idx.get_shape()[-1].value]),
            shape=[-1]), tf.reshape(tf.cast(idx, tf.int32), shape=[-1])], axis=1), shape=[batch_size, -1, 2])
        pooled_points_pl = tf.gather_nd(self.block_elm.points_pl,mask)

        if (is_subsampling):
            return pooled_points_pl,tf.gather_nd(network,mask)
        else:

            center_indexes = tf.argmin(tf.gather_nd(distances,mask),axis=1)

            mask = tf.reshape(tf.stack([tf.reshape(
                        tf.tile(temp, [1, center_indexes.get_shape()[-1].value]),shape=[-1]),
                        tf.reshape(tf.cast(center_indexes, tf.int32), shape=[-1])], axis=1),
                    shape=[batch_size, -1, 2])


            # Take max according to assignments, output is a K dim tensor
            pooled_network = tf.reduce_max(
                tf.multiply(tf.expand_dims(tf.cast(tf.equal(tf.expand_dims(tf.gather_nd(idx, mask),axis=1),tf.expand_dims(idx,axis=2)),tf.float32),axis=3),
                            tf.expand_dims(network, axis=1)),
                axis=2)

            return pooled_points_pl,pooled_network
