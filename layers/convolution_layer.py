import tensorflow as tf
import sys
sys.path.append('../')
import tf_util

class ConvLayer:
    def __init__(self, input_channels, block_elemnts, out_channels,scope,is_training,use_xavier=True,l2_regularizer = 1.0e-3):
        self.input_channels = input_channels
        self.block_elemnts = block_elemnts
        self.out_channels = out_channels
        self.scope = scope
        self.use_xavier = use_xavier
        self.is_training = is_training
        self.weight_decay = 0.0
        self.l2_regularizer = l2_regularizer

        with tf.variable_scope(self.scope) as sc:

            self.k_tensor = tf_util._variable_with_weight_decay('weights',
                                                           shape=[self.out_channels, self.input_channels, self.block_elemnts.num_of_translations],
                                                           use_xavier=self.use_xavier,
                                                           stddev=0.1, wd=self.weight_decay)

    def get_convlution_operator(self,functions_pl,interpolation,dtype=tf.float32):
        translations = self.block_elemnts.kernel_translations

        distances = self.block_elemnts.get_distance_matrix()
        points_translations_dot = tf.matmul(self.block_elemnts.points_pl, tf.transpose(translations, [0, 2, 1]))
        translations_square = tf.reduce_sum(translations * translations, axis=2)

        # Find weights w of the extension operator. result dimensions are BATCH_SIZE x NUM_OF_POINTS x  POINT_CLOUD_FUNCTION_DIM
        if (interpolation):
            w_tensor = tf.matrix_solve_ls(self.block_elemnts.get_interpolation_matrix(),
                                          functions_pl,l2_regularizer=self.l2_regularizer)
        else:
            w_tensor = tf.multiply(
                tf.expand_dims(tf.reciprocal(tf.reduce_sum(self.block_elemnts.get_interpolation_matrix(), axis=2)),
                               axis=2), functions_pl)

        # Calculate the product of w_tensor and the kernel weights. Result dimensions are TRANSLATIONS x BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        b_tensor = tf.transpose(tf.matmul(tf.tile(tf.expand_dims(w_tensor, axis=1), [1, self.out_channels, 1, 1]),
                                          tf.tile(tf.expand_dims(self.k_tensor, axis=0),
                                                  [self.block_elemnts.batch_size, 1, 1, 1])),
                                [3, 0, 2, 1])

        def convopeator_per_translation(input):
            translation_index = input[1]
            b_per_translation = input[0]

            dot = tf.tile(-2 * tf.slice(points_translations_dot, [0, 0, translation_index],
                                        [self.block_elemnts.batch_size, self.block_elemnts.num_of_points, 1]),
                          [1, 1, self.block_elemnts.num_of_points])

            q_tensor = tf.exp(-(tf.add(tf.add(distances,
                                              tf.add(dot, -tf.transpose(dot, [0, 2, 1]))),
                                       tf.expand_dims(tf.slice(translations_square, [0, translation_index],
                                                               [self.block_elemnts.batch_size, 1]), axis=2)))
                              / (2 * self.block_elemnts.combined_sigma ** 2))

            return tf.matmul(q_tensor, b_per_translation)

        # Calculate the pcnn convolution operator. Result dimensions are BATCH_SIZE x NUM_OF_POINTS x OUT_CHANNELS
        return tf.reduce_sum(tf.map_fn(convopeator_per_translation,
                                              elems=(b_tensor, tf.range(0, self.block_elemnts.num_of_translations)),
                                              dtype=dtype), axis=0)

    def get_layer(self, functions_pl, with_bn, bn_decay,interpolation,with_Relu = True):

        convlution_operation = self.get_convlution_operator(functions_pl,interpolation)

        with tf.variable_scope(self.scope) as sc:
            biases = tf_util._variable_on_cpu('biases', [self.out_channels],
                                              tf.constant_initializer(0.0))
            outputs = tf.nn.bias_add(convlution_operation, biases)

            if (with_bn):
                outputs = tf_util.batch_norm_template(outputs, self.is_training, 'bn', [0, 1], bn_decay)

            if (with_Relu):
                outputs = tf.nn.relu(outputs)
            return outputs