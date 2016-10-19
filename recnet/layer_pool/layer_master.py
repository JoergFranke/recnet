__author__ = 'joerg'

######                           Imports
########################################
from abc import ABCMeta, abstractmethod
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

class LayerMaster(object):

    __metaclass__ = ABCMeta


    ###### Abstract sequenceiteration method
    ########################################
    @abstractmethod
    def sequence_iteration(self):
        pass

    def sqr_ortho(self, rng, ndim):
        W = rng.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype(T.config.floatX)

    def rec_ortho(self, rng, ndim, ndim_factor):
        W = np.concatenate([self.sqr_ortho(rng, ndim) for i in xrange(ndim_factor)], axis=1)
        return W

    def rec_uniform_sqrt(self, rng, ndimA, ndimB):
        return rng.uniform(-np.sqrt(1./ndimB), np.sqrt(1./ndimB), (ndimA,ndimB))

    def rec_uniform_const(self, rng, ndimA, ndimB):
        return rng.uniform(-0.1,0.1, (ndimA, ndimB))

    def rec_normal_const(self, rng, ndimA, ndimB):
        return rng.normal(0, 2./(ndimA+ndimB), (ndimA, ndimB))

    def vec_uniform_sqrt(self, rng, ndim):
        return rng.uniform(-np.sqrt(1./ndim), np.sqrt(1./ndim), ndim)

    def vec_uniform_const(self, rng, ndim):
        return rng.uniform(-0.1,0.1, ndim)

    def vec_normal_const(self, rng, ndim):
        return  rng.normal(0, 1./(ndim), ndim)








