__author__ = 'joerg'


######                           Imports
########################################
import theano.tensor as T
import output_layer
import recurrent_layer







class SuperLayer:

    def __init__(self, rng,trng, prm_structure,prm_data, layer_no, old_weights=None):

        # Parameters
        rng = rng
        trng = trng

        n_in = prm_structure["net_size"][layer_no-1]
        n_out = prm_structure["net_size"][layer_no]
        unit_type = prm_structure["net_unit_type"][layer_no]
        activation_type = prm_structure["net_act_type"][layer_no]
        self.arch = prm_structure["net_arch"][layer_no]
        n_batches = prm_data["batch_size"]

        if unit_type in ["softmax",]:
            layer_ = getattr(output_layer, unit_type)
        elif unit_type in ["tanh", "ReLu", "LSTM", "LSTMnp", "GRU"]:
            layer_ = getattr(recurrent_layer, unit_type)

        if self.arch == "uni" or self.arch == "ff":
            if not old_weights == None:
                forward_weights = old_weights
                self.forward_layer = layer_(rng, trng, n_in, n_out, n_batches, forward_weights)
            else:
                self.forward_layer = layer_(rng, trng, n_in, n_out, n_batches)
        elif self.arch == "bi":

            if n_out % 2 != 0:
                raise Warning("Hidden layer number have to be even in case of bi")

            n_out = n_out / 2

            if not old_weights == None:
                forward_weights = old_weights[:old_weights.__len__() / 2]
                backward_weights = old_weights[old_weights.__len__() / 2:]

                self.forward_layer = layer_(rng, trng, n_in, n_out, n_batches, forward_weights)
                self.backward_layer = layer_(rng, trng, n_in, n_out, n_batches, backward_weights, go_backwards=True)
            else:
                self.forward_layer = layer_(rng, trng, n_in, n_out, n_batches)
                self.backward_layer = layer_(rng, trng, n_in, n_out, n_batches, go_backwards=True)
        else:
            raise Warning("No valid net architecture (uni or bi)")


    def sequence_iteration(self, in_seq, mask, use_dropout, dropout_value=1):

        if self.arch == "uni" or self.arch == "ff":
            out_sig = self.forward_layer.sequence_iteration(in_seq, mask, use_dropout, dropout_value)
            return (out_sig)
        elif self.arch == "bi":
            out_seq_f = self.forward_layer.sequence_iteration(in_seq, mask, use_dropout, dropout_value)
            out_seq_b = self.backward_layer.sequence_iteration(in_seq, mask, use_dropout, dropout_value)

            out_sig = T.concatenate([out_seq_f, out_seq_b[::-1]], axis=2)

            return (out_sig)

    @property
    def weights(self):
        if self.arch == "uni" or self.arch == "ff":
            return self.forward_layer.weights
        elif self.arch == "bi":
            return self.forward_layer.weights + self.backward_layer.weights




