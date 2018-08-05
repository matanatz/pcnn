import tensorflow as tf
import numpy as np
import sys
from pyhocon import ConfigFactory

sys.path.append('../layers')
from convlayer_elements import ConvElements
from convlution_layer import ConvLayer


class TestConvLayer(tf.test.TestCase):

    def run_test(self):
        I = 80
        J = 8
        L = 16
        M = 5
        sigma = np.sqrt(1 / I)
        ndtype = np.float64
        dtype = tf.float64
        X = np.random.rand(I, 3).astype(np.float64)
        F = np.random.rand(I, J).astype(np.float64)
        K = np.random.rand(J, M, L).astype(np.float64)
        T = sigma * np.random.rand(L, 3).astype(np.float64)

        from scipy.spatial import distance_matrix
        dist = np.power(distance_matrix(X,X),2)
        Phi = np.exp(-dist / (2 * sigma * sigma))
#       W = np.linalg.solve(Phi,F)

        W = np.reciprocal(np.expand_dims(np.sum(Phi,axis=1),1)) * F

        convOutput = np.zeros([I, M])
        for mm in range(M):
            for ii_tag in range(I):
                for jj in range(J):
                    for ll in range(L):
                        for ii in range(I):
                            gaussian = np.exp(-np.power(np.linalg.norm(X[ii_tag,:]-X[ii,:]-T[ll,:]) , 2) / (4 * sigma * sigma))
                            convOutput[ii_tag, mm] = convOutput[ii_tag, mm] + K[jj, mm, ll] * W[ii, jj] * gaussian

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            elm = ConvElements(tf.expand_dims(tf.constant(np.array(X),dtype=dtype), 0), (1. / np.sqrt(X.shape[0])).astype(ndtype), ndtype(1.0), ndtype(1.0))
            elm.kernel_translations = tf.expand_dims(tf.constant(np.array(T),dtype=dtype), 0)
            elm.num_of_translations = 16
            model = ConvLayer(8, elm, 5, 'test', tf.constant(False),l2_regularizer = 0.0)
            model.k_tensor = tf.constant(np.array(np.transpose(K,[1,0,2])),dtype=dtype)

            m = model.get_convlution_operator(tf.expand_dims(tf.constant(np.array(F),dtype=dtype), 0),False,dtype)
            local = sess.run(m)[0]
            print (local.shape)
            convOutput = np.array(convOutput)
            #print ('mean relative error : {0}'.format(np.abs(local - convOutput)/np.array(convOutput)))
            print (100 * np.linalg.norm((local - np.array(convOutput))/np.array(convOutput),np.inf))
            self.assertLess (np.linalg.norm((local - np.array(convOutput))/np.array(convOutput),np.inf),1.0e-6)

a = TestConvLayer()
a.run_test()