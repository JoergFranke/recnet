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
            pass
            #raise TypeError('rank mismatch between coding and true distributions')

    def output_error(self, input_sequence,   true_output,weight=None):
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

    def output_error(self, input_sequence,   true_output, weight=None):
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



######            CTC
########################################
class CTC():

    def normal_ctc(self, predict,Y):
        #Y_size= Y[-1] #todo rebuild
        labels = Y #[:Y_size]
        blank = labels[-1]
        n = labels.shape[0]

        labels2 = T.concatenate((labels, [blank, blank]))
        sec_diag = T.neq(labels2[:-2], labels2[2:]) * \
                   T.eq(labels2[1:-1], blank)

        recurrence_relation = \
            T.eye(n) + \
            T.eye(n, k=1) + \
            T.eye(n, k=2) * sec_diag.dimshuffle((0, 'x'))

        #predict2 = T.transpose(predict)
        #predict2.shape.eval()
        pred_y = predict[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, accum: curr * T.dot(accum, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.eye(n)[0]]
        )

        # TODO: -2 only if blank at end
        labels_probab = T.sum(probabilities[-1, -2:])
        return -T.log(labels_probab) # todo error ist extrem klein -> ctc_cost funktioniert nicht, eventuell pred.transpose


    def log_ctc(self, predict,Y):

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

        labels = Y #[:Y_size]
        blank = labels[-1]
        n = labels.shape[0]

        _1000 = T.eye(n)[0]
        prev_mask = 1 - _1000
        prevprev_mask = T.neq(labels[:-2], labels[2:]) * \
                        T.eq(labels[1:-1], blank)
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
        log_labels_probab = log_probs[-1, -1]
        return -log_labels_probab


    # def ctc_cost_batch(self, predict,Y):
    #
    #     labels = Y
    #     blank = labels[:,-1]
    #     max_n = labels.shape[1]
    #
    #     labels2 = T.concatenate((labels, [blank, blank]))
    #     sec_diag = T.neq(labels2[:-2], labels2[2:]) * \
    #                T.eq(labels2[1:-1], blank)
    #
    #     recurrence_relation = \
    #         T.eye(max_n) + \
    #         T.eye(max_n, k=1) + \
    #         T.eye(max_n, k=2) * sec_diag.dimshuffle((0, 'x'))
    #
    #     #predict2 = T.transpose(predict)
    #     #predict2.shape.eval()
    #     pred_y = predict[:,:, labels]
    #
    #     probabilities, _ = theano.scan(
    #         lambda curr, accum: curr * T.dot(accum, recurrence_relation),
    #         sequences=[pred_y],
    #         outputs_info=[T.eye(max_n)[0]]
    #     )
    #
    #     # TODO: -2 only if blank at end
    #     labels_probab = T.sum(probabilities[-1,:, -2:])
    #     return -T.log(labels_probab) # todo error ist extrem klein -> ctc_cost funktioniert nicht, eventuell pred.transpose
    #
    #
    #
    #
    # @staticmethod
    # def class_batch_to_labeling_batch(y, y_hat, y_hat_mask=None):
    #     #y_hat = y_hat * y_hat_mask.dimshuffle(0, 'x', 1)
    #     batch_size = y_hat.shape[2]
    #     res = y_hat[:, y.astype('int32'), T.arange(batch_size)]
    #     return res
    #
    # @staticmethod
    # def recurrence_relation(y, y_mask, blank_symbol):
    #     n_y = y.shape[0]
    #     blanks = T.zeros((2, y.shape[1])) + blank_symbol
    #     ybb = T.concatenate((y, blanks), axis=0).T
    #     sec_diag = (T.neq(ybb[:, :-2], ybb[:, 2:]) *
    #                 T.eq(ybb[:, 1:-1], blank_symbol) *
    #                 y_mask.T)
    #
    #     # r1: LxL
    #     # r2: LxL
    #     # r3: LxLxB
    #     r2 = T.eye(n_y, k=1)
    #     r3 = (T.eye(n_y, k=2).dimshuffle(0, 1, 'x') *
    #           sec_diag.dimshuffle(1, 'x', 0))
    #
    #     return r2, r3
    #
    # #@classmethod
    # def path_probabs(self, y, y_hat, y_mask, y_hat_mask, blank_symbol):
    #     #pred_y = self.class_batch_to_labeling_batch(y, y_hat, y_hat_mask)
    #     #r2, r3 = self.recurrence_relation(y, y_mask, blank_symbol)
    #
    #     batch_size = y_hat.shape[2]
    #     pred_y = y_hat[:, y.astype('int32'), T.arange(batch_size)]
    #
    #
    #     n_y = y.shape[0]
    #     blanks = T.zeros((2, y.shape[1])) + blank_symbol
    #     ybb = T.concatenate((y, blanks), axis=0).T
    #     sec_diag = (T.neq(ybb[:, :-2], ybb[:, 2:]) *
    #                 T.eq(ybb[:, 1:-1], blank_symbol) *
    #                 y_mask.T)
    #
    #     # r1: LxL
    #     # r2: LxL
    #     # r3: LxLxB
    #     r2 = T.eye(n_y, k=1)
    #     r3 = (T.eye(n_y, k=2).dimshuffle(0, 1, 'x') *
    #           sec_diag.dimshuffle(1, 'x', 0))
    #
    #
    #
    #     def step(p_curr, p_prev):
    #         # instead of dot product, we * first
    #         # and then sum oven one dimension.
    #         # objective: T.dot((p_prev)BxL, LxLxB)
    #         # solusion: Lx1xB * LxLxB --> LxLxB --> (sumover)xLxB
    #         dotproduct = (p_prev + T.dot(p_prev, r2) +
    #                       (p_prev.dimshuffle(1, 'x', 0) * r3).sum(axis=0).T)
    #         return p_curr.T * dotproduct * y_mask.T  # B x L
    #
    #     probabilities, _ = theano.scan(
    #         step,
    #         sequences=[pred_y],
    #         outputs_info=[T.eye(y.shape[0])[0] * T.ones(y.T.shape)])
    #     return probabilities

    def mb_normal_ctc(self, network_output,   true_output, mask):

        y = true_output.dimshuffle(1,0)
        y_hat = network_output.dimshuffle(0, 2, 1)

        mask = T.addbroadcast(mask, 2)
        y_hat_mask = mask.dimshuffle(0,1,)


        # T x C+1  B

        blank_symbol = y_hat.shape[1] - 1
        # blanked_y, blanked_y_mask = self.add_blanks(
        #     y=y,
        #     blank_symbol=num_classes.astype(floatX),
        #     y_mask=y_mask)

        y_mask = T.ones(y.shape)




        #y_hat_mask_len = T.sum(y_hat_mask, axis=0, dtype='int32')
        #y_mask_len = y.shape[0] # T.sum(y_mask, axis=0, dtype='int32')
        # probabilities = self.path_probabs(y, y_hat,
        #                                       y_mask, y_hat_mask,
        #                                       blank_symbol)

        batch_size = y_hat.shape[2]
        pred_y = y_hat[:, y.astype('int32'), T.arange(batch_size)]


        n_y = y.shape[0]
        blanks = T.zeros((2, y.shape[1])) + blank_symbol
        ybb = T.concatenate((y, blanks), axis=0).T
        sec_diag = (T.neq(ybb[:, :-2], ybb[:, 2:]) *
                    T.eq(ybb[:, 1:-1], blank_symbol) *
                    y_mask.T)

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
            return p_curr.T * dotproduct * y_mask.T  # B x L

        probabilities, _ = theano.scan(
            step,
            sequences=[pred_y],
            outputs_info=[T.eye(y.shape[0])[0] * T.ones(y.T.shape)])


        labels_probab = T.sum(probabilities[-1,:, -2:])
        return T.mean(-T.log(labels_probab))

        # batch_size = probabilities.shape[1]
        # labels_probab = (probabilities[y_hat_mask_len - 1,
        #                                T.arange(batch_size),
        #                                y_mask_len - 1] +
        #                  probabilities[y_hat_mask_len - 1,
        #                                T.arange(batch_size),
        #                                y_mask_len - 2])
        # avg_cost = T.mean(-T.log(labels_probab))
        # return avg_cost


    def output_error(self, network_output,   true_output, mask, weights=None):

        # blanked_y = true_output.dimshuffle(1,0)
        # y_hat = network_output.dimshuffle(0, 2, 1)
        #
        # mask = T.addbroadcast(mask, 2)
        # y_hat_mask = mask.dimshuffle(0,1,)
        #
        #
        # # T x C+1  B
        #
        # num_classes = y_hat.shape[1] - 1
        # # blanked_y, blanked_y_mask = self.add_blanks(
        # #     y=y,
        # #     blank_symbol=num_classes.astype(floatX),
        # #     y_mask=y_mask)
        #
        # blanked_y_mask = T.ones(blanked_y.shape)
        #
        # final_cost = self.cost(blanked_y, y_hat,
        #                            blanked_y_mask, y_hat_mask,
        #                            num_classes)
        #return final_cost

################################################################### up is new

        #batch_size = network_output.shape[1]
        #network_output = T.reshape(network_output, [network_output.shape[1],network_output.shape[2],network_output.shape[0]])

        #cost = self.ctc_cost(network_output, true_output[0,:]) #todo rebuild 2, batch size 1, no batch dimension


        #cost = self.mb_normal_ctc(network_output,   true_output, mask)

        cost = self.log_ctc(network_output[:,0,:], true_output[0,:])

        #cost = self.normal_ctc(network_output[:,0,:], true_output[0,:])
        #cost2 = self.ctc_cost(network_output[:,1,:], true_output[1,:])
        #cost = (cost1 + cost2 )/ 2.

        return cost #todo because softmax change



