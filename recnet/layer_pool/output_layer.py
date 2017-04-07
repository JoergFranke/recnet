from __future__ import absolute_import, print_function, division
"""
This file contains output layers.
"""

######                           Imports
########################################
import numpy as np
import theano
import theano.tensor as T
from .layer_master import LayerMaster


######                     Softmax Layer
########################################
class softmax(LayerMaster):
    def __init__(self, rng, trng, n_in, n_out, n_batches=None, activation=None, old_weights=None): #self, rng,trng, prm_structure, layer_no, old_weights=None):

        # Parameters
        self.n_in = n_in
        self.n_out = n_out

        #w_out_np2 = self.rec_uniform_sqrt(rng, self.n_in, self.n_out)

        #w_out_np2 = 1 * (rng.rand(self.n_in, self.n_out) - 0.5)
        #b_out_np2 = 1 * (rng.rand(self.n_out) - 0.5)

        w_out_np2 = rng.uniform(-np.sqrt(1./self.n_in), np.sqrt(1./self.n_in), (self.n_in, self.n_out))
        b_out_np2 = rng.uniform(-np.sqrt(1./self.n_in), np.sqrt(1./self.n_in), self.n_out)

        #w_out_np2 = 0.01 * rng.randn(self.n_in, self.n_out)
        #b_out_np2 = np.ones(self.n_out)

        # todo initialization

        if old_weights == None:
            self.t_w_out = theano.shared(name='w_out', value=w_out_np2.astype(T.config.floatX))
            self.t_b_out = theano.shared(name='b_out', value=b_out_np2.astype(T.config.floatX))
        else:
            self.t_w_out = theano.shared(name='w_out', value=old_weights[0].astype(T.config.floatX))
            self.t_b_out = theano.shared(name='b_out', value=old_weights[1].astype(T.config.floatX))


        self.trng = trng

        # All layer weights
        self.weights = [self.t_w_out,self.t_b_out]


    def sequence_iteration(self, output, mask,use_dropout=0,dropout_value=0.5):

        dot_product = T.dot(output , self.t_w_out)

        net_o = T.add( dot_product , self.t_b_out )

        ex_net = T.exp(net_o)
        sum_net = T.sum(ex_net, axis=2, keepdims=True)
        softmax_o = ex_net / sum_net


        mask = T.addbroadcast(mask, 2) # to do nesseccary?
        output = T.mul(mask, softmax_o)   + T.mul( (1. - mask) , 1e-6 )

        return output #result

######                     Linear Layer
########################################
class linear(LayerMaster):
    def __init__(self, rng, trng, n_in, n_out, n_batches=None, activation=None,
                 old_weights=None):  # self, rng,trng, prm_structure, layer_no, old_weights=None):

        # Parameters
        self.n_in = n_in
        self.n_out = n_out

        # w_out_np2 = self.rec_uniform_sqrt(rng, self.n_in, self.n_out)

        # w_out_np2 = 1 * (rng.rand(self.n_in, self.n_out) - 0.5)
        # b_out_np2 = 1 * (rng.rand(self.n_out) - 0.5)

        w_out_np2 = rng.uniform(-np.sqrt(1. / self.n_in), np.sqrt(1. / self.n_in), (self.n_in, self.n_out))
        b_out_np2 = rng.uniform(-np.sqrt(1. / self.n_in), np.sqrt(1. / self.n_in), self.n_out)

        # w_out_np2 = 0.01 * rng.randn(self.n_in, self.n_out)
        # b_out_np2 = np.ones(self.n_out)

        # todo initialization

        if old_weights == None:
            self.t_w_out = theano.shared(name='w_out', value=w_out_np2.astype(T.config.floatX))
            self.t_b_out = theano.shared(name='b_out', value=b_out_np2.astype(T.config.floatX))
        else:
            self.t_w_out = theano.shared(name='w_out', value=old_weights[0].astype(T.config.floatX))
            self.t_b_out = theano.shared(name='b_out', value=old_weights[1].astype(T.config.floatX))

        self.trng = trng

        # All layer weights
        self.weights = [self.t_w_out, self.t_b_out]

    def sequence_iteration(self, output, mask, use_dropout=0, dropout_value=0.5):

        dot_product = T.dot(output, self.t_w_out)

        linear_o = T.add(dot_product, self.t_b_out)


        mask = T.addbroadcast(mask, 2)  # to do nesseccary?
        output = T.mul(mask, linear_o) + T.mul((1. - mask), 1e-6)

        return output  # result


### TEST FUNCTIONS # to do make new file with test functions
from scipy.stats import multivariate_normal
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

if __name__ == "__main__":

    x, y = np.mgrid[-1:1:.05, -1:1:.2]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    multimormal = rv.pdf(pos)
    example_output = np.empty([40,2,10])
    example_output[:,0,:] = multimormal
    example_output[:,1,:] = 1- multimormal

    mask = np.ones([40,2,1])
    #mask[38:,:,:] = np.zeros([2,2,1]) todo test mask

    def np_softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    # theano test part
    rng = np.random.RandomState(123)
    trng = RandomStreams(123)

    n_in = 10
    n_out = 5

    t_sig =  T.tensor3('t_sig', dtype=theano.config.floatX)
    t_mask = T.tensor3('t_mask', dtype=theano.config.floatX)

    layer_class = softmax(rng, trng, n_in, n_out)

    t_rslt = layer_class.sequence_iteration(t_sig, t_mask)

    tfn = theano.function([t_sig,t_mask],t_rslt)

    softmax_output = tfn(example_output.astype(theano.config.floatX), mask.astype(theano.config.floatX))

    w_out = layer_class.weights[0].eval()
    b_out = layer_class.weights[1].eval()

    correct_output = np.empty([40,2,5])
    for b in range(2):
        for s in range(40):
            act_sig = np.dot(example_output[s,b,:] ,w_out) + b_out
            correct_output[s,b,:] = np_softmax(act_sig)

    print(np.max(correct_output - softmax_output) )