__author__ = 'Joerg Franke'
"""
This file contains the implementation of different recurrent layers.
"""

######                           Imports
########################################

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

######            Network Initialization
########################################
#Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks."
# Aistats. Vol. 9. 2010.
class layerMaster:

    def ortho_weight(self, rng, ndim):
        W = rng.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype(T.config.floatX)

    def init_ortho_weight_matrix(self, rng, ndim):
        W = np.concatenate([self.ortho_weight(rng, ndim),
                               self.ortho_weight(rng, ndim),
                               self.ortho_weight(rng, ndim),
                               self.ortho_weight(rng, ndim)], axis=1)
        #return W
        ##return rng.uniform(-np.sqrt(1./n_out), np.sqrt(1./n_out), (n_out, 4*n_out))
        return rng.uniform(-0.1, 0.1, (ndim, 4*ndim))

    def init_peephole_weight(self, rng, ndim):
        #return rng.uniform(-0.1,0.1, ndim)
        return rng.uniform(-np.sqrt(1./ndim), np.sqrt(1./ndim), ndim)

    def init_input_weight(self, rng, n_in, n_out):
        ##return rng.normal(0, 2./(n_in+n_out), (n_in, n_out))
        #return rng.uniform(-np.sqrt(1./n_in), np.sqrt(1./n_in), (n_in, n_out))
        return rng.uniform(-0.1, 0.1, (n_in, n_out))



######                        LSTM Layer
########################################
class LSTMlayer(layerMaster):
    """
    Long short term memory layer

    key ideas of implementation:
        - peepholes at input gate and forget gate but not at output gate
        - calculate dot product of input and input weights before scan function
        - calculate dot product of previous output and weights only ones per sequence
        - weights and biases separate
        - one function for each step, one for each sequence
    """

    def __init__(self, rng, trng, n_in, n_out, n_batches, old_weights=None,
                 go_backwards=False):  # , prm_structure, layer_no ):

        # Parameters
        self.go_backwards = go_backwards

        # Random
        self.rng = rng
        self.trng = trng

        if old_weights == None:

            np_weights = OrderedDict()
            # Peephole weights (input- forget- output- gate)
            np_weights['w_ig_c'] = self.init_peephole_weight(self.rng, n_out)
            np_weights['w_fg_c'] = self.init_peephole_weight(self.rng,
                                                             n_out) + 2  # Forgot gate with +2 initialized for keeping sequences right from begin
            np_weights['w_og_c'] = self.init_peephole_weight(self.rng, n_out)
            # Previous output weights
            np_weights['w_ifco'] = self.init_ortho_weight_matrix(rng, n_out)
            np_weights['b_ifco'] = np.zeros(4 * n_out)
            # Input weights
            np_weights['w_ifco_x'] = self.init_input_weight(rng, n_in, 4 * n_out)
            np_weights['b_ifco_x'] = np.zeros(4 * n_out)

            self.weights = []
            for kk, pp in np_weights.items():
                self.weights.append(theano.shared(name=kk, value=pp.astype(T.config.floatX)))

        # load old weights
        else:
            self.weights = []
            for pp in old_weights:
                self.weights.append(theano.shared(value=pp.astype(T.config.floatX)))

        # Init last output and cell state
        ol_t00_np1 = np.zeros([n_batches, n_out]).astype(dtype=theano.config.floatX)
        cs_t00_np1 = np.zeros([n_batches, n_out]).astype(dtype=theano.config.floatX)
        self.t_ol_t00 = theano.shared(name='ol_b_t00', value=ol_t00_np1.astype(T.config.floatX))
        self.t_cs_t00 = theano.shared(name='cs_b_t00', value=cs_t00_np1.astype(T.config.floatX))

        # Outputs & cell states
        self.t_o = T.matrix('ol', dtype=theano.config.floatX)
        self.t_cs = T.vector('cs', dtype=theano.config.floatX)

    def t_forward_step(self, mask, cur_w_in_sig, pre_out_sig, pre_cell_sig, w_ig_c, w_fg_c, w_og_c, w_ifco, b_ifco,
                       t_n_out):

        ifco = T.add(T.dot(pre_out_sig, w_ifco), b_ifco)

        inner_act = T.tanh  # T.nnet.hard_sigmoid #T.tanh # T.nnet.hard_sigmoid T.tanh
        gate_act = T.nnet.hard_sigmoid  # T.nnet.hard_sigmoid #T.nnet.sigmoid

        # Input Gate
        ig_t1 = gate_act(T.add(ifco[:, 0:t_n_out], T.mul(pre_cell_sig, w_ig_c), cur_w_in_sig[:, 0:t_n_out]))
        # Forget Gate
        fg_t1 = gate_act(T.add(ifco[:, 1 * t_n_out:2 * t_n_out], T.mul(pre_cell_sig, w_fg_c),
                               cur_w_in_sig[:, 1 * t_n_out:2 * t_n_out]))
        # Cell State
        cs_t1 = T.add(T.mul(fg_t1, pre_cell_sig), T.mul(ig_t1, inner_act(
            T.add(ifco[:, 2 * t_n_out:3 * t_n_out], cur_w_in_sig[:, 2 * t_n_out:3 * t_n_out]))))

        mask = T.addbroadcast(mask, 1)
        cs_t1 = mask * cs_t1 + (1. - mask) * pre_cell_sig
        # functionality: cs_t1 =   T.switch(mask , cs_t1, pre_cell_sig)

        # Output Gate
        og_t1 = gate_act(
            T.add(ifco[:, 3 * t_n_out:4 * t_n_out], T.mul(cs_t1, w_og_c), cur_w_in_sig[:, 3 * t_n_out:4 * t_n_out]))
        # Output LSTM
        out_sig = T.mul(og_t1, inner_act(cs_t1))

        out_sig = mask * out_sig + (1. - mask) * pre_out_sig

        return [out_sig, cs_t1]

    def sequence_iteration(self, in_seq, mask, use_dropout, dropout_value=1):

        in_seq_d = T.switch(use_dropout,
                            (in_seq *
                             self.trng.binomial(in_seq.shape,
                                                p=dropout_value, n=1,
                                                dtype=in_seq.dtype)),
                            in_seq)

        w_in_seq = T.add(T.dot(in_seq_d, self.weights[5]), self.weights[6])
        t_n_out = self.weights[4].shape[0] / 4

        [out_seq, cell_seq], updates = theano.scan(
            fn=self.t_forward_step,
            sequences=[mask, w_in_seq],
            outputs_info=[self.t_ol_t00, self.t_cs_t00],
            non_sequences=self.weights[:5] + [t_n_out],
            go_backwards=self.go_backwards,
            truncate_gradient=-1,
            # n_steps=50,
            strict=True,
            allow_gc=False,
        )

        return (out_seq)

######                        GRU Layer
########################################
class GRUlayer(layerMaster):
    """
    Gated recurrent unit layer

    key ideas of implementation:
    """

    def __init__(self, rng, trng, n_in, n_out, n_batches, old_weights=None,go_backwards=False): #, prm_structure, layer_no ):

        # Parameters
        self.go_backwards = go_backwards

        # Random
        self.rng = rng
        self.trng = trng

        if old_weights == None:

            np_weights = OrderedDict()

            # reset gate
            np_weights['w_r'] =  rng.uniform(-0.1, 0.1, (n_in, n_out))
            np_weights['u_r'] = rng.uniform(-0.1, 0.1, (n_out, n_out))
            np_weights['b_r'] =  np.zeros(n_out) #rng.uniform(-0.1, 0.1, (1, n_out))
            # update gate
            np_weights['w_z'] = rng.uniform(-0.1, 0.1, (n_in, n_out))
            np_weights['u_z'] = rng.uniform(-0.1, 0.1, (n_out, n_out))
            np_weights['b_z'] = np.zeros(n_out) #rng.uniform(-0.1, 0.1, (1, n_out))
            # update weights
            np_weights['w_up'] = rng.uniform(-0.1, 0.1, (n_in, n_out))
            np_weights['u_up'] = rng.uniform(-0.1, 0.1, (n_out, n_out))
            np_weights['b_up'] = np.zeros(n_out) #rng.uniform(-0.1, 0.1, (1, n_out))

            self.weights = []
            for kk, pp in np_weights.items():
                self.weights.append(theano.shared(name=kk, value=pp.astype(T.config.floatX)))

        # load old weights
        else:
            self.weights = []
            for pp in old_weights:
                self.weights.append(theano.shared(value=pp.astype(T.config.floatX)))

        #Init last output and cell state
        ol_t00_np1 = np.zeros([n_batches,n_out]).astype(dtype=theano.config.floatX)
        #cs_t00_np1 = np.zeros([n_batches,n_out]).astype(dtype=theano.config.floatX)
        self.t_ol_t00 = theano.shared(name='ol_b_t00', value=ol_t00_np1.astype(T.config.floatX))
        #self.t_cs_t00 = theano.shared(name='cs_b_t00', value=cs_t00_np1.astype(T.config.floatX))

        #Outputs & cell states
        #self.t_o = T.matrix('ol', dtype=theano.config.floatX)
        #self.t_cs = T.vector('cs', dtype=theano.config.floatX)




    def t_forward_step(self,mask, in_sig, h_pre, w_r, u_r,b_r, w_z, u_z, b_z, w_up, u_up, b_up):



        signal_act = T.tanh
        gate_act = T.nnet.hard_sigmoid


        r = gate_act( T.add( T.add( T.dot(in_sig, w_r) , T.dot( h_pre, u_r) ) , b_r ))
        z = gate_act( T.add( T.add( T.dot(in_sig, w_z) , T.dot(h_pre, u_z) ), b_z))

        h_update = signal_act( T.add( T.add( T.dot(in_sig, w_up) , T.dot( T.mul( h_pre, r), u_up) ), b_up))

        h_new = T.add( (1.-z) * h_pre , z * h_update )

        mask = T.addbroadcast(mask, 1)
        out_sig =  T.add( mask * h_new   , (1. - mask) * h_pre )

        return out_sig


    def sequence_iteration(self, in_seq, mask, use_dropout,dropout_value=1):

        in_seq_d = T.switch(use_dropout,
                             (in_seq *
                              self.trng.binomial(in_seq.shape,
                                            p=dropout_value, n=1,
                                            dtype=in_seq.dtype)),
                             in_seq)

        out_seq, updates = theano.scan(
                                                fn=self.t_forward_step,
                                                sequences=[mask, in_seq_d],
                                                outputs_info=[self.t_ol_t00],
                                                non_sequences=[i for i in self.weights], #+[t_n_out],
                                                go_backwards = self.go_backwards,
                                                truncate_gradient=-1,
                                                #n_steps=50,
                                                strict=True,
                                                allow_gc=False,
                                                )

        return(out_seq)

######      LSTM without peepholes Layer
########################################
class LSTMnPlayer(layerMaster):
    """
    Long short term memory layer without peepholes

    key ideas of implementation:
        - calculate dot product of input and input weights before scan function
        - calculate dot product of previous output and weights only ones per sequence
        - weights and biases separate
        - one function for each step, one for each sequence
    """

    def __init__(self, rng, trng, n_in, n_out, n_batches, old_weights=None,
                 go_backwards=False):  # , prm_structure, layer_no ):

        # Parameters
        self.go_backwards = go_backwards

        # Random
        self.rng = rng
        self.trng = trng

        if old_weights == None:

            np_weights = OrderedDict()

            # Previous output weights
            np_weights['w_ifco'] = self.init_ortho_weight_matrix(rng, n_out)
            np_weights['b_ifco'] = np.zeros(4 * n_out)
            # Input weights
            np_weights['w_ifco_x'] = self.init_input_weight(rng, n_in, 4 * n_out)
            np_weights['b_ifco_x'] = np.zeros(4 * n_out)

            self.weights = []
            for kk, pp in np_weights.items():
                self.weights.append(theano.shared(name=kk, value=pp.astype(T.config.floatX)))

        # load old weights
        else:
            self.weights = []
            for pp in old_weights:
                self.weights.append(theano.shared(value=pp.astype(T.config.floatX)))

        # Init last output and cell state
        ol_t00_np1 = np.zeros([n_batches, n_out]).astype(dtype=theano.config.floatX)
        cs_t00_np1 = np.zeros([n_batches, n_out]).astype(dtype=theano.config.floatX)
        self.t_ol_t00 = theano.shared(name='ol_b_t00', value=ol_t00_np1.astype(T.config.floatX))
        self.t_cs_t00 = theano.shared(name='cs_b_t00', value=cs_t00_np1.astype(T.config.floatX))

        # Outputs & cell states
        self.t_o = T.matrix('ol', dtype=theano.config.floatX)
        self.t_cs = T.vector('cs', dtype=theano.config.floatX)

    def t_forward_step(self, mask, cur_w_in_sig, pre_out_sig, pre_cell_sig, w_ifco, b_ifco,
                       t_n_out):

        ifco = T.add(T.dot(pre_out_sig, w_ifco), b_ifco)

        inner_act = T.tanh  # T.nnet.hard_sigmoid #T.tanh # T.nnet.hard_sigmoid T.tanh
        gate_act = T.nnet.hard_sigmoid  # T.nnet.hard_sigmoid #T.nnet.sigmoid

        # Input Gate
        ig_t1 = gate_act(T.add(ifco[:, 0:t_n_out], cur_w_in_sig[:, 0:t_n_out]))
        # Forget Gate
        fg_t1 = gate_act(T.add(ifco[:, 1 * t_n_out:2 * t_n_out],
                               cur_w_in_sig[:, 1 * t_n_out:2 * t_n_out]))
        # Cell State
        cs_t1 = T.add(T.mul(fg_t1, pre_cell_sig), T.mul(ig_t1, inner_act(
            T.add(ifco[:, 2 * t_n_out:3 * t_n_out], cur_w_in_sig[:, 2 * t_n_out:3 * t_n_out]))))

        mask = T.addbroadcast(mask, 1)
        cs_t1 = mask * cs_t1 + (1. - mask) * pre_cell_sig
        # functionality: cs_t1 =   T.switch(mask , cs_t1, pre_cell_sig)

        # Output Gate
        og_t1 = gate_act(
            T.add(ifco[:, 3 * t_n_out:4 * t_n_out], cur_w_in_sig[:, 3 * t_n_out:4 * t_n_out]))
        # Output LSTM
        out_sig = T.mul(og_t1, inner_act(cs_t1))

        out_sig = mask * out_sig + (1. - mask) * pre_out_sig

        return [out_sig, cs_t1]

    def sequence_iteration(self, in_seq, mask, use_dropout, dropout_value=1):

        in_seq_d = T.switch(use_dropout,
                            (in_seq *
                             self.trng.binomial(in_seq.shape,
                                                p=dropout_value, n=1,
                                                dtype=in_seq.dtype)),
                            in_seq)

        w_in_seq = T.add(T.dot(in_seq_d, self.weights[2]), self.weights[3])
        t_n_out = self.weights[1].shape[0] / 4

        [out_seq, cell_seq], updates = theano.scan(
            fn=self.t_forward_step,
            sequences=[mask, w_in_seq],
            outputs_info=[self.t_ol_t00, self.t_cs_t00],
            non_sequences=self.weights[:2] + [t_n_out],
            go_backwards=self.go_backwards,
            truncate_gradient=-1,
            # n_steps=50,
            strict=True,
            allow_gc=False,
        )

        return (out_seq)


######          Bidirectional LSTM Layer
########################################
class BLSTMlayer(layerMaster):

    def __init__(self, rng, trng, n_in, n_out, n_batches, old_weights=None):

        # Number of forward/backward hidden nodes
        assert n_out % 2 == 0, "Hidden layer number have to be even in case of BLSTM"

        n_out = n_out / 2

        if not old_weights == None:
            forward_weights = old_weights[:old_weights.__len__()/2]
            backward_weights = old_weights[old_weights.__len__()/2:]

            self.forward_layer  = LSTMlayer(rng, trng, n_in, n_out, n_batches, forward_weights, False)
            self.backward_layer = LSTMlayer(rng, trng, n_in, n_out, n_batches, backward_weights, True)
        else:
            self.forward_layer  = LSTMlayer(rng, trng, n_in, n_out, n_batches)
            self.backward_layer = LSTMlayer(rng, trng, n_in, n_out, n_batches, go_backwards=True)

        self.weights = self.forward_layer.weights + self.backward_layer.weights

    def sequence_iteration(self, in_seq,mask, use_dropout,dropout_value=1):

        out_seq_f = self.forward_layer.sequence_iteration(in_seq,mask, use_dropout,dropout_value)
        out_seq_b = self.backward_layer.sequence_iteration(in_seq,mask, use_dropout,dropout_value)

        out_sig = T.concatenate([out_seq_f,out_seq_b[::-1]], axis=2)

        return(out_sig)



######                     Softmax Layer
########################################
class softmaxLayer(layerMaster):
    def __init__(self, rng,trng, prm_structure, layer_no, old_weights=None):

        # Parameters
        self.n_in = prm_structure["net_size"][layer_no-1:layer_no][0]
        self.n_out = prm_structure["net_size"][layer_no:layer_no+1][0]

        #output layer
        w_out_np2 = self.init_input_weight(rng, self.n_in, self.n_out)
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


    def vanilla(self, output):
        net_o = T.add( T.dot(output , self.t_w_out) , self.t_b_out )
        result, updates = theano.map(T.nnet.softmax, net_o)
        return result

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

    def softmax(self, output, mask,use_dropout=0,dropout_value=0.5):
        prm = [self.t_w_out, self.t_b_out, use_dropout,dropout_value]
        result, updates = theano.map(self._drop_out_softmax, [output, mask], prm)
        return result