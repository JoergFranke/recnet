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

    # def recurrence_relation(self, size):
    #     big_I = T.eye(size+2, dtype=theano.config.floatX)
    #
    #     b = (T.arange(size) % 2)
    #     b = T.cast(b, 'float32')
    #     mat = T.eye(size, dtype=theano.config.floatX) + big_I[2:,1:-1] + T.mul( big_I[2:,:-2], b)
    #     return mat

    # def recurrence_relation(self, size, Y):
    #     labels2 = T.concatenate((Y, [Y[-1], Y[-1]]))
    #     sec_diag = T.neq(labels2[:-2], labels2[2:]) * \
    #                T.eq(labels2[1:-1], Y[-1])
    #
    #     recurrence_relation = \
    #         T.eye(size) + \
    #         T.eye(size, k=1) + \
    #         T.eye(size, k=2) * sec_diag.dimshuffle((0, 'x'))
    #     return recurrence_relation
    #
    # def step(self, p_curr,p_prev, rr):
    #         # print(p_curr.dtype)
    #         # print(rr.dtype)
    #         # print(p_prev.dtype)
    #
    #         return p_curr * T.dot(p_prev,rr)
    #
    # def path_probs(self, predict,Y, Y_size):
    #     print(predict.shape)
    #     print(Y.shape)
    #     P = predict[:,Y]
    #     #a = theano.shared(Y.shape[0], dtype=theano.config.floatX)
    #     ##a =
    #     rr = self.recurrence_relation(Y_size, Y)
    #
    #     probs,_ = theano.scan(
    #             self.step,
    #             sequences = [P],
    #             outputs_info = [T.eye(Y_size, dtype=theano.config.floatX)[0]],
    #             non_sequences=[rr],
    #         )
    #     return probs
    #
    # def ctc_cost(self, predict,Y):
    #     Y_size= Y[-1]
    #     #print(Y_size.dtype)
    #     Y = Y[:Y_size]
    #     forward_probs  = self.path_probs(predict,Y, Y_size)
    #     backward_probs = self.path_probs(predict[::-1],Y[::-1], Y_size)[::-1,::-1]
    #     probs = forward_probs * backward_probs / predict[:,Y]
    #     total_prob = T.sum(probs)
    #
    #     #total_prob = T.sum(forward_probs[-1, -2:])
    #
    #     return -T.log(total_prob)

    def ctc_cost(self, predict,Y):
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

        predict = T.transpose(predict)
        pred_y = predict[:, labels]

        probabilities, _ = theano.scan(
            lambda curr, accum: curr * T.dot(accum, recurrence_relation),
            sequences=[pred_y],
            outputs_info=[T.eye(n)[0]]
        )

        # TODO: -2 only if blank at end
        labels_probab = T.sum(probabilities[-1, -2:])
        return -T.log(labels_probab) # todo error ist extrem klein -> ctc_cost funktioniert nicht, eventuell pred.transpose



    def output_error(self, network_output,   true_output, weights=None):

                #batch_size = network_output.shape[1]
        #network_output = T.reshape(network_output, [network_output.shape[1],network_output.shape[2],network_output.shape[0]])

        cost = self.ctc_cost(network_output, true_output[0,:])
        return cost #todo because softmax change

        # #batch_size = network_output.shape[1]
        # network_output = T.reshape(network_output, [network_output.shape[1],network_output.shape[2],network_output.shape[0]])
        #
        # cost = self.ctc_cost(network_output[0,:,:], true_output[0,:])
        # return cost


        # cost, _ = theano.map(
        #     fn=self.ctc_cost,
        #     sequences=[network_output, true_output]
        #
        # )
        # #for i in range(batch_size):
        # #    cost[i] = self.ctc_cost(network_output[i, :T.sum(mask[:,i,0]),:],true_output[:true_output[-1]])
        #
        #
        # # outputs, updates = theano.scan(
        # #                                 fn=self._crossentropy,
        # #                                 sequences=[input_sequence, true_output],
        # #                                 )
        # #print(cost.type)
        # #print(cost[0].type)
        #
        # return cost[0] #T.mean(cost)

