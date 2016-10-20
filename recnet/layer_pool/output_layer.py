__author__ = 'Joerg Franke'


######                           Imports
########################################
import numpy as np
import theano
import theano.tensor as T


from layer_master import LayerMaster


######                     Softmax Layer
########################################
class softmax(LayerMaster):
    def __init__(self, rng, trng, n_in, n_out, n_batches=None, activation=None, old_weights=None): #self, rng,trng, prm_structure, layer_no, old_weights=None):

        # Parameters
        self.n_in = n_in
        self.n_out = n_out

        #output layer
        w_out_np2 = self.rec_uniform_sqrt(rng, self.n_in, self.n_out)
        b_out_np2 = np.zeros(self.n_out)


        if old_weights == None:
            self.t_w_out = theano.shared(name='w_out', value=w_out_np2.astype(T.config.floatX))
            self.t_b_out = theano.shared(name='b_out', value=b_out_np2.astype(T.config.floatX))
        else:
            self.t_w_out = theano.shared(name='w_out', value=old_weights[0].astype(T.config.floatX))
            self.t_b_out = theano.shared(name='b_out', value=old_weights[1].astype(T.config.floatX))


        self.trng = trng

        # All layer weights
        self.weights = [self.t_w_out,self.t_b_out]


    # def vanilla(self, output):
    #     net_o = T.add( T.dot(output , self.t_w_out) , self.t_b_out )
    #     result, updates = theano.map(T.nnet.softmax, net_o)
    #     return result

    def _drop_out_softmax(self, signal, mask,t_w_out,t_b_out, use_dropout, dropout_value):
        d_w_out = T.switch(use_dropout,
                             (t_w_out *
                              self.trng.binomial(t_w_out.shape,
                                            p=dropout_value, n=1,
                                            dtype=t_w_out.dtype)),
                             t_w_out)

        net_o = T.add( T.dot(signal , d_w_out) , t_b_out)
        output = T.nnet.softmax(net_o)

        mask = T.addbroadcast(mask, 1)
        output = mask * output   + (1. - mask) * 0.1**8

        return output

    def sequence_iteration(self, output, mask,use_dropout=0,dropout_value=0.5):
        prm = [self.t_w_out, self.t_b_out, use_dropout,dropout_value]
        result, updates = theano.map(self._drop_out_softmax, [output, mask], prm)
        return result