__author__ = 'Joerg Franke'
"""
This file contains different error/loss functions.
"""

######                           Imports
########################################
import theano
import theano.tensor as T


######    2-class weightes cross entropy
########################################
class w2_cross_entropy():
    @staticmethod
    def _w_crossentropy(coding_dist, true_dist,weight):
        if true_dist.ndim == coding_dist.ndim:
            no_bound =  true_dist[:,:,0] *  T.log(coding_dist[:,:,0])
            bound =  true_dist[:,:,1] *  T.log(coding_dist[:,:,1]) * weight
            return - (no_bound + bound)
        else:
            raise TypeError('rank mismatch between coding and true distributions')

    def output_error(self, input_sequence,   true_output,weight):
        return T.mean(self._w_crossentropy(input_sequence, true_output,weight))



###### dynamic 2-class weightes cross entropy
########################################
class dynamic_cross_entropy():
    @staticmethod
    def _w_crossentropy(self, coding_dist, true_dist):
        if true_dist.ndim == coding_dist.ndim:
            no_bound =  true_dist[:,:,0] *  T.log(coding_dist[:,:,0])

            weight =  0.5 / ( T.sum(true_dist[:,:,1]) / (true_dist.shape[0] * true_dist.shape[1]) )
            bound =  true_dist[:,:,1] *  T.log(coding_dist[:,:,1]) * weight
            return - (no_bound + bound)

        else:
            raise TypeError('rank mismatch between coding and true distributions')

    def output_error(self, input_sequence,   true_output,weight=None):
        return T.mean(self._w_crossentropy(input_sequence, true_output))


######            Standard cross entropy
########################################
class cross_entropy():
    @staticmethod
    def _crossentropy(coding_dist, true_dist):
        if true_dist.ndim == coding_dist.ndim:
            return T.nnet.categorical_crossentropy(coding_dist, true_dist)
        else:
            raise TypeError('rank mismatch between coding and true distributions')

    def output_error(self, input_sequence,   true_output, weights=None):

        outputs, updates = theano.scan(
                                        fn=self._crossentropy,
                                        sequences=[input_sequence, true_output],
                                        )
        return T.mean(outputs)

