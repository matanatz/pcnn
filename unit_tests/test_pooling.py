import tensorflow as tf
import numpy as np
import sys
from scipy.spatial import distance_matrix

sys.path.append('../layers')
from pooling import PoolingLayer
from convlayer_elements import ConvElements


class TestPooling(tf.test.TestCase):

    def run_test(self):
        with self.test_session():

            B = 2
            I = 8
            I_tag = int(I / 2)
            J = 3

            X = np.random.rand(B,I, 3)

            F = np.random.rand(B,I, J)

            idx, startidx = self.fps(X, B, I_tag, I)

            X_tag = X[[[[x] for x in list(range(B))]] + [idx.tolist()]]

            poolOut = np.zeros([B,I_tag, J])
            for b in range(B):
                dist = np.power(distance_matrix(X[b], X[b]), 2)


                for ii in range(I_tag):
                    for jj  in range(J):

                        vals = []
                        for kk in range(I):
                            a = idx[b]

                            minidx = np.argmin(dist[kk, a])
                            if (minidx == ii):
                                vals = vals + [F[b,kk, jj]]

                        poolOut[b,ii, jj] = np.max(vals)

            print(poolOut.shape)
            print (poolOut)

            elm = ConvElements(tf.constant(X,dtype=tf.float32),np.sqrt(I,dtype=np.float32),np.float32(1),np.float32(1))
            m = PoolingLayer(elm,J,J,I_tag).get_layer(tf.constant(F,dtype=tf.float32),use_fps=tf.constant(True,dtype=tf.bool),startidx=startidx)

            self.assertLess (np.linalg.norm(m[1].eval() - poolOut),1.0e-6)
            self.assertLess (np.linalg.norm(m[0].eval() - X_tag),1.0e-6)

    def fps(self,points,B,I_tag,I):
        r = np.sum(points * points, 2)
        r = np.expand_dims(r, axis=2)
        distance = r - 2 * np.matmul(points, np.transpose(points, [0, 2, 1])) + np.transpose(r, [0, 2, 1])
        temp = [[x] for x in list(range(B))]
        idx = np.random.randint(0, I)
        startidx = idx

        if (I_tag > 1):
            gather = np.argmax(distance[:, idx, :], axis=1)
            idx = np.stack([np.array(B * [idx]), gather], axis=1)

            for step in range(2, max(I_tag, 2)):
                gather = distance[[temp] + [idx.tolist()]]

                idx = np.concatenate([idx, np.expand_dims(np.argmax(np.min(gather, 1), 1), 1)], 1)
        else:
            idx = np.array(B * [idx])
        return idx,startidx


TestPooling().run_test()