__author__ = 'Joerg Franke'
"""
This file contains the rnnModel-Class which builds the RNN-model and the training, validation and forward functions.
It connects the layers, adds regularization and optimizations.
"""

######                           Imports
########################################
#import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
import numpy as np
from collections import OrderedDict

import error_function
import update_function
from model_master import ModelMaster
from layer_pool.super_layer import SuperLayer


######                     Theano Config
########################################
import theano
import theano.tensor as T
#theano.config.device='gpu0'
#theano.config.floatX = 'float32'
#theano.config.mode = 'FAST_RUN'
#theano.config.optimizer = 'fast_run'
#theano.config.allow_gc = False

#theano.config.lib.cnmem =1
theano.config.scan.allow_gc = False
#theano.config.optimizer_excluding ='low_memory'
#theano.config.scan.allow_output_prealloc = True
#theano.config.exception_verbosity='high'


class rnnModel(ModelMaster):


    def build_model(self, old_weights=None):


        ######            Create model variables
        ########################################
        self.X_tv2 = T.tensor3('X_tv2', dtype=theano.config.floatX)
        if self.prm.optimize["loss_function"] == "CTC":
            self.Y_tv2 = T.imatrix('Y_tv2', )
        else:
            self.Y_tv2 = T.tensor3('Y_tv2', dtype=theano.config.floatX)
        self.M_tv2 = T.tensor3('M_tv2', dtype=theano.config.floatX)
        self.X_tv2_v = T.tensor3('Y_tv2', dtype=theano.config.floatX)


        ######           Create model parameters
        ########################################
        tpo = OrderedDict() #theano optimization parameter
        for key, value in self.prm.optimize.items():
            if not isinstance(value,str):
                tpo[key] = theano.shared(name=key, value=np.asarray(value, dtype=theano.config.floatX))


        ###### Add noise to input if noisy_input
        ########################################
        if self.prm.optimize["noisy_input"]:
            self.X_tv2 = self.X_tv2 + self.trng.normal(size=self.X_tv2.shape, avg=0, std=tpo["noise_level"])
        else:
            self.X_tv2 = self.X_tv2


        ######                     Create layers
        ########################################
        network_layer = []
        for i in np.arange(1, self.prm.struct["hidden_layer"]+2):
            network_layer.append(SuperLayer(self.rng,self.trng, self.prm.struct,self.prm.data, i, old_weights[i-1]))

        self.layer_weights = [l.weights for l in network_layer] # + [output_layer.weights]
        self.all_weights = sum([l for l in self.layer_weights],[])


        ######             Choose error function
        ########################################
        try:
            loss_function_ = getattr(error_function, self.prm.optimize["loss_function"])
        except AttributeError:
            raise NotImplementedError("Class `{}` does not implement `{}`".format(error_function.__name__, self.prm.optimize["loss_function"]))

        loss_function = loss_function_() #w2_cross_entropy() #cross_entropy() #


        ######                    Connect layers
        ########################################
        ## training part
        t_signal = []
        t_signal.append(self.X_tv2)
        for l in range(self.prm.struct["hidden_layer"]+1):
            t_signal.append(network_layer[l].sequence_iteration(t_signal[l],self.M_tv2, tpo["use_dropout"],tpo["dropout_level"]))
            if self.prm.struct["identity_func"] and l >= 1 and l <= self.prm.struct["hidden_layer"]:
                t_signal[l+1] = t_signal[l+1] + t_signal[l]
        self.t_net_out = t_signal[-1]
        o_error = loss_function.output_error(self.t_net_out,  self.Y_tv2, tpo["bound_weight"])

        ## validation/test part
        v_signal = []
        v_signal.append(self.X_tv2_v)
        for l in range(self.prm.struct["hidden_layer"]+1):
            v_signal.append(network_layer[l].sequence_iteration(v_signal[l],self.M_tv2, use_dropout=0))
            if self.prm.struct["identity_func"] and l >= 1 and l <= self.prm.struct["hidden_layer"]:
                v_signal[l+1] = v_signal[l+1] + v_signal[l]
        self.v_net_out = v_signal[-1]
        self.v_error = loss_function.output_error(self.v_net_out,  self.Y_tv2 , tpo["bound_weight"])


        ######                    Regularization
        ########################################
        def regularization(weights):
            w_error = 0
            for w in weights:
                w_error = w_error + self.prm.optimize["reg_factor"]  * T.square(T.mean(T.sqr(w)))
            return w_error

        if self.prm.optimize["regularization"]:
            self.t_error = o_error + regularization(self.all_weights)
        else:
            self.t_error = o_error


        ######                    Call optimizer
        ########################################
        try:
            optimizer_ = getattr(update_function, self.prm.optimize["optimization"])
        except AttributeError:
            raise NotImplementedError("Class `{}` does not implement `{}`".format(update_function.__name__, self.prm.optimize["optimization"]))

        optimizer = optimizer_(self.rng, self.all_weights)

        self.updates = optimizer.fit(self.all_weights,self.t_error, tpo)

        #self.gradients = optimizer.grads todo remove
    ######            BUILD THEANO FUNCTIONS
    ########################################
    def get_training_function(self):
        train_fn = theano.function(inputs=[self.X_tv2, self.Y_tv2, self.M_tv2],
                                   outputs=[self.t_net_out, self.t_error],
                                   updates=self.updates,
                                   allow_input_downcast=True
                                   #mode='DebugMode',
                                   #profile=profile,
                                   #mode=theano.Mode(linker='c'),
                                    )
        return train_fn

    def get_validation_function(self):
        valid_fn = theano.function(inputs=[self.X_tv2_v,self.Y_tv2, self.M_tv2],
                                        outputs=[self.v_net_out, self.v_error],
                                        allow_input_downcast=True
                                    )
        return valid_fn

    def get_forward_function(self):
        forward_fn = theano.function(inputs=[self.X_tv2_v,self.M_tv2],
                                          outputs=[self.v_net_out, ],
                                          allow_input_downcast=True
                                    )
        return forward_fn