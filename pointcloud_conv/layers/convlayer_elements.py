import tensorflow as tf
import numpy as np


class ConvElements:
    def __init__(self, points_pl ,sigma,spacing,kernel_sigma_factor):

        self.points_pl = points_pl
        self.batch_size = self.points_pl.get_shape()[0].value
        self.num_of_points = self.points_pl.get_shape()[1].value
        self.sigma_f = sigma
        self.sigma_k =  kernel_sigma_factor * self.sigma_f
        self.spacing = spacing
        self.combined_sigma =  tf.sqrt(self.sigma_f * self.sigma_f + self.sigma_k * self.sigma_k)
        self.num_of_centers = 3
        self.num_of_translations = self.num_of_centers ** 3
        self.kernel_translations = self.get_kernel_translations(spacing)

    def get_kernel_translations(self,spacing):
        grid = tf.linspace(-spacing * self.sigma_k, spacing * self.sigma_k, tf.constant(3))

        return tf.tile(tf.reshape(tf.transpose(tf.reshape(tf.stack(tf.meshgrid(grid, grid, grid)), [3, -1])),
                                            [1, self.num_of_translations, 3]), [self.batch_size, 1, 1])

    def get_conv_matrix(self):
        c1 = tf.tile(tf.reshape(tf.square(tf.norm(self.points_pl, axis=2)), [self.batch_size,self.num_of_points, 1, 1]),
                     [1, 1, self.num_of_points, self.num_of_translations])
        c2 = tf.tile(tf.reshape(tf.square(tf.norm(self.points_pl, axis=2)), [self.batch_size,1, self.num_of_points, 1]),
                     [1, self.num_of_points, 1, self.num_of_translations])
        c3 = tf.tile(tf.reshape(tf.square(tf.norm(self.kernel_translations, axis=2)), [self.batch_size,1, 1, self.num_of_translations]),
                     [1, self.num_of_points, self.num_of_points, 1])

        c4 = tf.tile(-2.0 * tf.expand_dims(tf.matmul(self.points_pl, tf.transpose(self.points_pl,[0, 2, 1])),
                                       dim=3), [1,1, 1, self.num_of_translations])
        c5 = tf.tile(-2.0 * tf.expand_dims(tf.matmul(self.points_pl, tf.transpose(self.kernel_translations,[0,2,1])),
                                       dim=2), [1, 1, self.num_of_points, 1])
        c6 = tf.tile(2.0 * tf.expand_dims(tf.matmul(self.points_pl, tf.transpose(self.kernel_translations,[0, 2, 1])),
                                          dim=1), [1, self.num_of_points, 1, 1])

        C_add = c1 + c2 + c3 + c4 + c5 + c6


        C = tf.exp(-C_add / (2.0 * self.combined_sigma * self.combined_sigma))

        return C

    def get_interpolation_matrix(self):
        return tf.exp(-self.get_distance_matrix() / (2.0 * self.sigma_f * self.sigma_f))

    def get_distance_matrix(self):
        r = tf.reduce_sum(self.points_pl * self.points_pl, 2)
        r = tf.expand_dims(r, dim=2)
        D = r - 2 * tf.matmul(self.points_pl, tf.transpose(self.points_pl, [0, 2, 1])) + tf.transpose(r, [0, 2, 1])
        return D

