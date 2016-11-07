__author__ = 'Joerg Franke'
"""
This file contains different error/loss functions.
"""

######                           Imports
########################################
from abc import ABCMeta, abstractmethod
import theano
import theano.tensor as T



######              loss function master
########################################
class LossMaster():    
    __metaclass__ = ABCMeta
    
    def __init__(self, tpo, batch_size):
        self.tpo = tpo
        self.batch_size = batch_size

        if batch_size > 1:
            if type(self).__name__ in ['CTC', 'CTClog']:
                raise Warning("Please use for batch size > 1 mbCTC or mbCTClog")

        
    

######    2-class weightes cross entropy
########################################
class w2_cross_entropy(LossMaster):

    def _w_crossentropy(self, coding_dist, true_dist):

        no_bound =  true_dist[:,:,0] *  T.log(coding_dist[:,:,0])
        bound =  true_dist[:,:,1] *  T.log(coding_dist[:,:,1]) * self.tpo["bound_weight"]
        return - (no_bound + bound)


    def output_error(self, input_sequence,   true_output, mask):

        outputs = self._w_crossentropy(input_sequence, true_output)

        outputs = T.mul(outputs.dimshuffle(0,1,'x'), mask)

        return T.mean(outputs)



######            Standard cross entropy
########################################
class cross_entropy(LossMaster):

    def output_error(self, input_sequence,   true_output, mask):

        outputs = T.nnet.categorical_crossentropy(input_sequence, true_output)

        outputs = T.mul(outputs.dimshuffle(0,1,'x'), mask)

        return T.mean(outputs)



"""
Connectionist temporal classification
Referece: Graves, Alex, et al. "Connectionist temporal classification:
labelling unsegmented sequence data with recurrent neural networks."
Proceedings of the 23rd international conference on Machine learning.
ACM, 2006.
Credits: Shawn Tan, Rakesh Var, Mohammad Pezeshki
"""

######                 CTC loss function
########################################
class CTC(LossMaster):

    def _ctc_normal(self, predict,labels):

        n = labels.shape[0]

        labels2 = T.concatenate((labels, [self.tpo["CTC_blank"], self.tpo["CTC_blank"]]))
        sec_diag = T.neq(labels2[:-2], labels2[2:]) * \
                   T.eq(labels2[1:-1], self.tpo["CTC_blank"])

        recurrence_relation = \
            T.eye(n) + \
            T.eye(n, k=1) + \
            T.eye(n, k=2) * sec_diag.dimshuffle((0, 'x'))

        pred_y = predict[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, accum: curr * T.dot(accum, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.eye(n)[0]]
        )

        labels_probab = T.sum(probabilities[-1, -2:])
        return -T.log(labels_probab)

    def output_error(self, network_output,   true_output, mask=None):

        cost = self._ctc_normal(network_output[:,0,:], true_output[0,:])

        return cost


######             log CTC loss function
########################################
class CTClog(LossMaster):

    def _ctc_log(self, predict,labels):

        def safe_log(x):
            return T.log(T.maximum(x, 1e-20).astype(theano.config.floatX))

        def safe_exp(x):
            return T.exp(T.minimum(x, 1e20).astype(theano.config.floatX))

        def logadd_simple(x, y):
            return x + safe_log(1 + safe_exp(y - x))

        def logadd(x, y, *zs):
            sum = logadd_simple(x, y)
            for z in zs:
                sum = logadd_simple(sum, z)
            return sum

        def logmul(x, y):
            return x + y

        n = labels.shape[0]

        _1000 = T.eye(n)[0]
        prev_mask = 1 - _1000
        prevprev_mask = T.neq(labels[:-2], labels[2:]) * \
                        T.eq(labels[1:-1], self.tpo["CTC_blank"])
        prevprev_mask = T.concatenate(([0, 0], prevprev_mask))
        prev_mask = safe_log(prev_mask)
        prevprev_mask = safe_log(prevprev_mask)
        prev = T.arange(-1, n-1)
        prevprev = T.arange(-2, n-2)
        log_pred_y = T.log(predict[:, labels])

        def step(curr, accum):
            return logmul(curr,
                          logadd(accum,
                                 logmul(prev_mask, accum[prev]),
                                 logmul(prevprev_mask, accum[prevprev])))

        log_probs, _ = theano.scan(
            step,
            sequences=[log_pred_y],
            outputs_info=[safe_log(_1000)]
        )

        # TODO: Add -2 if n > 1 and blank at end
        log_labels_probab = log_probs[-1, -1] #T.sum(log_probs[-1, -2:]) to do
        return -log_labels_probab

    def output_error(self, network_output,   true_output, mask=None):

        cost = self._ctc_log(network_output[:,0,:], true_output[0,:])

        return cost


###### Mini Batch normal CTC loss function
########################################
class mbCTC(LossMaster):

    def __init__(self, tpo, batch_size):

        super(mbCTC, self).__init__(tpo, batch_size)
        self.blanks = T.zeros((2, self.tpo["batch_size"])) + self.tpo["CTC_blank"]

    def _mb_normal_ctc(self, network_output,   labels, mask):


        n_y = labels.shape[1] / 2
        y = labels[:,:n_y]
        y = y.dimshuffle(1,0)
        y_mask = labels[:,n_y:].astype(theano.config.floatX)

        # y_row = labels.dimshuffle(1,0)
        # n_y = y_row.shape[0] / 2
        # y = y_row[:n_y,:]
        # y_mask = y_row[n_y:,:].astype(theano.config.floatX)

        y_hat = network_output.dimshuffle(0, 2, 1)

        pred_y = y_hat[:, y.astype('int32'), T.arange(self.tpo["batch_size"])]


        ybb = T.concatenate((y, self.blanks), axis=0).T
        sec_diag = (T.neq(ybb[:, :-2], ybb[:, 2:]) *
                    T.eq(ybb[:, 1:-1], self.tpo["CTC_blank"]) *
                    y_mask)



        # r1: LxL
        # r2: LxL
        # r3: LxLxB
        r2 = T.eye(n_y, k=1)
        r3 = (T.eye(n_y, k=2).dimshuffle(0, 1, 'x') *
              sec_diag.dimshuffle(1, 'x', 0))



        def step(p_curr, p_prev):
            # instead of dot product, we * first
            # and then sum oven one dimension.
            # objective: T.dot((p_prev)BxL, LxLxB)
            # solusion: Lx1xB * LxLxB --> LxLxB --> (sumover)xLxB
            dotproduct = (p_prev + T.dot(p_prev, r2) +
                          (p_prev.dimshuffle(1, 'x', 0) * r3).sum(axis=0).T)
            return p_curr.T * dotproduct * y_mask  # B x L

        probabilities, _ = theano.scan(
            step,
            sequences=[pred_y],
            outputs_info=[T.eye(n_y)[0] * T.ones([self.tpo["batch_size"], n_y])])


        labels_probab = T.sum(probabilities[-1,:, -2:])
        return T.mean(-T.log(labels_probab))


    def output_error(self, network_output,   labels, mask):

        cost = self._mb_normal_ctc(network_output,   labels, mask)

        return cost



######  Mini Batch log CTC loss function
########################################
class mbCTClog(LossMaster):

    def __init__(self, tpo, batch_size):

        super(mbCTClog, self).__init__(tpo, batch_size)
        self.blanks = T.zeros((2, self.tpo["batch_size"])) + self.tpo["CTC_blank"]


    @staticmethod
    def _epslog(x):
        return T.cast(T.log(T.clip(x, 1E-12, 1E12)),
                           theano.config.floatX)


    @staticmethod
    def log_add(a, b):
        max_ = T.maximum(a, b)
        return (max_ + T.log1p(T.exp(a + b - 2 * max_)))

    @staticmethod
    def log_dot_matrix(x, z):
        inf = 1E12
        log_dot = T.dot(x, z)
        zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
        return log_dot + zeros_to_minus_inf

    @staticmethod
    def log_dot_T(x, z):
        inf = 1E12
        log_dot = (x.dimshuffle(1, 'x', 0) * z).sum(axis=0).T
        zeros_to_minus_inf = (z.max(axis=0) - 1) * inf
        return log_dot + zeros_to_minus_inf.T


    def _mb_log_ctc(self, network_output,   labels, mask):




        #y_row = labels.dimshuffle(1,0)
        n_y = labels.shape[1] / 2
        y = labels[:,:n_y]
        y = y.dimshuffle(1,0)
        y_mask = labels[:,n_y:].astype(theano.config.floatX)

        y_hat = network_output.dimshuffle(0, 2, 1)

        pred_y = y_hat[:, y.astype('int32'), T.arange(self.tpo["batch_size"])]


        ybb = T.concatenate((y, self.blanks), axis=0).T
        sec_diag = (T.neq(ybb[:, :-2], ybb[:, 2:]) *
                    T.eq(ybb[:, 1:-1], self.tpo["CTC_blank"]) *
                    y_mask)

        r2 = T.eye(n_y, k=1)
        r3 = (T.eye(n_y, k=2).dimshuffle(0, 1, 'x') *
              sec_diag.dimshuffle(1, 'x', 0))


        def step(log_p_curr, log_p_prev):
            p1 = log_p_prev
            p2 = self.log_dot_matrix(p1, r2)
            p3 = self.log_dot_T(p1, r3)
            p123 = self.log_add(p3, self.log_add(p1, p2))

            return (log_p_curr.T +
                    p123 +
                    self._epslog(y_mask))

        log_probabs, _ = theano.scan(
            step,
            sequences=[self._epslog(pred_y)],
            outputs_info=[self._epslog(T.eye(n_y)[0] *
                                      T.ones([self.tpo["batch_size"], n_y]))])

        labels_probab = T.sum(log_probabs[-1,:, -2:])
        return T.mean(-labels_probab)




    def output_error(self, network_output,   labels, mask):

        cost = self._mb_log_ctc(network_output,   labels, mask)

        return cost

