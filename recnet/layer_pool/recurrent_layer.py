__author__ = 'joerg'
"""
This file contains the implementation of different recurrent layers.
"""




######                           Imports
########################################
from abc import ABCMeta, abstractmethod
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from layer_master import LayerMaster


######      Conventional recurrent layer
########################################
class conv(LayerMaster):
    """
    Hyperbolic tangent or rectified linear unit layer
    """

    def __init__(self, rng, trng, n_in, n_out, n_batches, activation, old_weights=None,go_backwards=False):  # , prm_structure, layer_no ):

        # Parameters
        self.go_backwards = go_backwards
        self.activation = activation

        # Random
        self.rng = rng
        self.trng = trng

        if old_weights == None:

            np_weights = OrderedDict()

            np_weights['w_in_hidden'] = self.rec_uniform_sqrt(rng, n_in, n_out)
            np_weights['w_hidden_hidden'] = self.sqr_ortho(rng, n_out)
            np_weights['b_act'] = np.zeros(n_out)

            self.weights = []
            for kk, pp in np_weights.items():
                self.weights.append(theano.shared(name=kk, value=pp.astype(T.config.floatX)))

        # load old weights
        else:
            self.weights = []
            for pp in old_weights:
                self.weights.append(theano.shared(value=pp.astype(T.config.floatX)))

        # Init last output and cell state
        init_hidden = np.zeros([n_batches, n_out]).astype(dtype=theano.config.floatX)
        self.t_init_hidden = theano.shared(name='init_hidden', value=init_hidden.astype(T.config.floatX))

    def t_forward_step(self, mask, cur_w_in_sig, pre_out_sig, w_hidden_hidden, b_act):

        pre_w_sig = T.dot(pre_out_sig, w_hidden_hidden)
        inner_act = self.activation
        out_sig = inner_act(T.add(cur_w_in_sig, pre_w_sig, b_act))

        mask = T.addbroadcast(mask, 1)
        out_sig_m = mask * out_sig + (1. - mask) * pre_out_sig
        return [out_sig_m]

    def sequence_iteration(self, in_seq, mask, use_dropout, dropout_value=1):

        in_seq_d = T.switch(use_dropout,
                            (in_seq *
                             self.trng.binomial(in_seq.shape,
                                                p=dropout_value, n=1,
                                                dtype=in_seq.dtype)),
                            in_seq)

        w_in_seq = T.dot(in_seq_d, self.weights[0])

        out_seq, updates = theano.scan(
                                        fn=self.t_forward_step,
                                        sequences=[mask, w_in_seq],
                                        outputs_info=[self.t_init_hidden],
                                        non_sequences=self.weights[1:],
                                        go_backwards=self.go_backwards,
                                        truncate_gradient=-1,
                                        # n_steps=50,
                                        strict=True,
                                        allow_gc=False,
        )
        return out_seq


######                        LSTM Layer
########################################
class LSTM(LayerMaster):
    """
    Long short term memory layer

    key ideas of implementation:
        - peepholes at input gate and forget gate but not at output gate
        - calculate dot product of input and input weights before scan function
        - calculate dot product of previous output and weights only ones per sequence
        - weights and biases separate
        - one function for each step, one for each sequence
    """

    def __init__(self, rng, trng, n_in, n_out, n_batches, activation, old_weights=None,
                 go_backwards=False):  # , prm_structure, layer_no ):

        # Parameters
        self.go_backwards = go_backwards
        self.activation = activation

        # Random
        self.rng = rng
        self.trng = trng

        if old_weights == None:

            np_weights = OrderedDict()
            # Peephole weights (input- forget- output- gate)
            np_weights['w_ig_c'] = self.vec_uniform_sqrt(self.rng, n_out)
            np_weights['w_fg_c'] = self.vec_uniform_sqrt(self.rng,
                                                             n_out) + 2  # Forgot gate with +2 initialized for keeping sequences right from begin
            np_weights['w_og_c'] = self.vec_uniform_sqrt(self.rng, n_out)
            # Previous output weights
            np_weights['w_ifco'] = self.rec_ortho(rng, n_out, 4)
            np_weights['b_ifco'] = np.zeros(4 * n_out)
            # Input weights
            np_weights['w_ifco_x'] = self.rec_uniform_sqrt(rng, n_in, 4 * n_out)
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


    def t_forward_step(self, mask, cur_w_in_sig, pre_out_sig, pre_cell_sig, w_ig_c, w_fg_c, w_og_c, w_ifco, b_ifco,
                       t_n_out):

        ifco = T.add(T.dot(pre_out_sig, w_ifco), b_ifco)

        inner_act = self.activation  # T.nnet.hard_sigmoid #T.tanh # T.nnet.hard_sigmoid T.tanh
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
        t_n_out = self.weights[6].shape[0] / 4

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

        return out_seq


######      LSTM without peepholes Layer
########################################
class LSTMnp(LayerMaster):
    """
    Long short term memory layer without peepholes

    key ideas of implementation:
        - calculate dot product of input and input weights before scan function
        - calculate dot product of previous output and weights only ones per sequence
        - weights and biases separate
        - one function for each step, one for each sequence
    """

    def __init__(self, rng, trng, n_in, n_out, n_batches, activation, old_weights=None,
                 go_backwards=False):  # , prm_structure, layer_no ):

        # Parameters
        self.go_backwards = go_backwards
        self.activation = activation

        # Random
        self.rng = rng
        self.trng = trng

        if old_weights == None:

            np_weights = OrderedDict()

            # Previous output weights
            np_weights['w_ifco'] = self.rec_ortho(rng, n_out, 4)
            np_weights['b_ifco'] = np.zeros(4 * n_out)
            # Input weights
            np_weights['w_ifco_x'] = self.rec_uniform_sqrt(rng, n_in, 4 * n_out)
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

        inner_act = self.activation # T.nnet.hard_sigmoid #T.tanh # T.nnet.hard_sigmoid T.tanh
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

        return out_seq


######                        GRU Layer
########################################
class GRU(LayerMaster):
    """
    Gated recurrent unit layer

    key ideas of implementation:
    """

    def __init__(self, rng, trng, n_in, n_out, n_batches, activation, old_weights=None,go_backwards=False): #, prm_structure, layer_no ):

        # Parameters
        self.go_backwards = go_backwards
        self.activation = activation

        # Random
        self.rng = rng
        self.trng = trng

        if old_weights == None:

            np_weights = OrderedDict()

            # Input weights for reset/update gate and update weights
            np_weights['w_rzup'] = self.rec_uniform_sqrt(rng,n_in, 3 * n_out ) # rng.uniform(-0.1, 0.1,(n_in, 3 * n_out))
            np_weights['b_rzup'] = np.zeros(3 * n_out)

            # reset and update gate
            np_weights['u_rz'] = self.rec_ortho(rng, n_out, 2) #self.uniform(-0.1, 0.1, (n_out, n_out))

            # reset gate
            #np_weights['u_r'] = self.sqr_ortho(rng, n_out) #self.uniform(-0.1, 0.1, (n_out, n_out))

            # update gate
            #np_weights['u_z'] = self.sqr_ortho(rng, n_out) #rng.uniform(-0.1, 0.1, (n_out, n_out))

            # update weights
            np_weights['u_up'] = self.sqr_ortho(rng, n_out) #rng.uniform(-0.1, 0.1, (n_out, n_out))


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
        self.t_ol_t00 = theano.shared(name='ol_b_t00', value=ol_t00_np1.astype(T.config.floatX))

    def t_forward_step(self,mask, rzup_in_sig, h_pre, u_rz, u_up, t_n_out): #u_r, u_z,



        signal_act = self.activation
        gate_act = T.nnet.sigmoid #T.nnet.hard_sigmoid

        preact = T.dot( h_pre, u_rz)


        r = gate_act( T.add( rzup_in_sig[:, 0:t_n_out] , preact[:, 0:t_n_out] )) #T.dot( h_pre, u_r) ) )
        z = gate_act( T.add( rzup_in_sig[:, t_n_out:2 * t_n_out] , preact[:, t_n_out:2 * t_n_out] )) #T.dot(h_pre, u_z) ))

        h_update = signal_act( T.add( rzup_in_sig[:, 2*t_n_out:3*t_n_out] , T.dot( T.mul( h_pre, r), u_up) ))

        h_new = T.add( (1.-z) * h_update , z * h_pre )

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

        rz_in_seq =  T.add( T.dot(in_seq_d, self.weights[0]) , self.weights[1] )
        t_n_out = self.weights[1].shape[0] / 3

        out_seq, updates = theano.scan(
                                        fn=self.t_forward_step,
                                        sequences=[mask, rz_in_seq],  # in_seq_d],
                                        outputs_info=[self.t_ol_t00],
                                        non_sequences=[i for i in self.weights][2:] + [t_n_out],
                                        go_backwards = self.go_backwards,
                                        truncate_gradient=-1,
                                        #n_steps=50,
                                        strict=True,
                                        allow_gc=False,
                                        )

        return out_seq