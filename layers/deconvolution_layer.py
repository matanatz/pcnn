import tensorflow as tf

class DeconvLayer:
    def __init__(self,points_from,functions_from,points_to):
        self.points_from = points_from
        self.functions_from = functions_from
        self.points_to = points_to

    def get_distance_between_points(self,points_from,points_to):
        r_to = tf.reduce_sum(points_to * points_to, 2)
        r_to = tf.expand_dims(r_to, dim=1)

        r_from = tf.reduce_sum(points_from * points_from, 2)
        r_from = tf.expand_dims(r_from, dim=2)

        return r_from - 2 * tf.matmul(points_from, tf.transpose(points_to, [0, 2, 1])) + r_to

    def get_layer(self):

        sigma = tf.reciprocal(tf.sqrt(1.0 * self.points_from.get_shape()[1].value))
        D = self.get_distance_between_points(self.points_from, self.points_from)
        intr = tf.exp(-D / (2.0 * sigma * sigma))

        w_tensor = tf.multiply(tf.expand_dims(tf.reciprocal(tf.reduce_sum(intr,axis=2)),axis=2),self.functions_from)

        D = self.get_distance_between_points(self.points_to,self.points_from)
        intr =  tf.exp(-D / (2.0 * sigma * sigma))

        return tf.matmul(intr,w_tensor)


