#!/usr/bin/env python

""" Little timer task """
"""___________________"""
"""    TRAIN MODEL
"""


######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False"
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags


######         IMPORTS          ######
import numpy as np
import sklearn.metrics
import time
from past.builtins import xrange
from collections import OrderedDict
from recnet.build_model import rnnModel
#import recnet


### 2. Step: Create new model
rn = rnnModel()
#rn = recnet.rnnModel(parameter)


### 1. Step: Define parameters
#parameter = OrderedDict()
rn.parameter["train_data_name"] = "little-timer_train.klepto"
rn.parameter["valid_data_name"] = "little-timer_valid.klepto"
rn.parameter["data_location"] = "data_set/"
rn.parameter["batch_size" ] = 10

rn.parameter["net_size"      ] = [      2,     10,         2]
rn.parameter["net_unit_type" ] = ['input', 'GRU', 'softmax']
rn.parameter["net_act_type"  ] = [    '-',  'tanh',      '-']
rn.parameter["net_arch"      ] = [    '-',    'bi',     'ff']

rn.parameter["epochs"        ] = 5
rn.parameter["learn_rate"    ] = 0.001
rn.parameter["use_dropout"   ] = False       # False, True
rn.parameter["regularization"] = False       # False, L2,  L1
rn.parameter["momentum_rate"] = 0.1
rn.parameter["decay_rate"] = 0.9
rn.parameter["optimization"  ] = "nesterov_momentum"  # sgd, nm_rmsprop, rmsprop, nesterov_momentum, adadelta
rn.parameter["loss_function" ] = "cross_entropy" # w2_cross_entropy, cross_entropy


rn.create(['train', 'valid'])

### 3. Step: Build model functions
#train_fn    = rn.get_training_function()
#valid_fn    = rn.get_validation_function()


### 4. Step: Train model
rn.pub("Start training")

### 4.1: Create minibatches for validation set
valid_mb_set_x, valid_mb_set_y, valid_mb_set_m = rn.get_mini_batches("valid")

### 4.2: Start epoch loop
for i in xrange(rn.prm.optimize["epochs"]):
    rn.pub("------------------------------------------")
    rn.pub(str(i)+" Epoch, Training run")
    time_0 = time.time()
    time_1 = time.time()
    ### 4.3: Create minibatches for training set
    mb_train_x, mb_train_y, mb_mask = rn.get_mini_batches("train")

    ### 4.4: Iterate over mini batches
    train_error = np.zeros(rn.prm.data["train_set_len" ]) # todo change to get_...
    for j in xrange(rn.prm.data["train_batch_quantity"]):

        ### 4.5: Train with one mini batch
        net_out, train_error[j] = rn.train_fn( mb_train_x[j],
                                            mb_train_y[j],
                                            mb_mask[j]
                                            )

        ### 4.6: Print training error
        if ( j%50) == 0 :
            rn.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_0) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_0 = time.time()

        ### 4.7: Print validation error
        if ( (j%500) == 0 or j == rn.prm.data["train_set_len" ]-1 ) and j>0:
            rn.pub("###########################################")
            rn.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([rn.prm.data["valid_batch_quantity"]])
            ce_error = np.zeros([rn.prm.data["valid_batch_quantity"]*rn.prm.data["batch_size"]])
            auc_error = np.zeros([rn.prm.data["valid_batch_quantity"]*rn.prm.data["batch_size"]])

            for v in np.arange(0,rn.prm.data["valid_batch_quantity"]):
                v_net_out_, v_error[v] = rn.valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])

                for b in np.arange(0,rn.prm.data["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]

                    count = v * rn.prm.data["batch_size"] + b

                    ce_error[count] = sklearn.metrics.log_loss( true_out,code_out)
                    auc_error[count] = sklearn.metrics.roc_auc_score( true_out,code_out)

            rn.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error)))
            rn.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
            rn.pub("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error)))
            rn.pub("###########################################")

            ### 4.8: Save current model
            rn.dump()

    rn.pub("###########################################")
    rn.pub("Insample Error: " + str(np.mean(train_error)))
    rn.pub("Epoch training duration: "+ str(time.time()-time_1) + "sec")

#Finale
rn.pub("## ||||||||||||||||||||||||||||||||||||||||")


