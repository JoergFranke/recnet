#!/usr/bin/env python

""" Little timer task """
"""___________________"""
"""    TRAIN MODEL
"""


######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False" #ast_run
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags


######         IMPORTS          ######
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import sklearn.metrics
import time
from collections import OrderedDict

from rnnfwk.build_model import rnnModel
from rnnfwk.data_handler import load_minibatches


###### GLOBAL TIMER
time_0 = time.time()


########## RANDOM STREAMS
prm_optimization = OrderedDict()
prm_optimization["seed"] = 211
rng = np.random.RandomState(prm_optimization["seed"])
trng = RandomStreams(prm_optimization["seed"] )


###### DATA IN
print("# Load data")
prm_structure = OrderedDict()
prm_structure["batch_size" ] = 10
prm_structure["corpus_name"] = "little-timer"
prm_structure["data_location"] = "data_set/"

data_name = prm_structure["corpus_name" ] + '_train'
train_mb_set_x,train_mb_set_y,train_mb_set_m = load_minibatches(prm_structure["data_location"], data_name, prm_structure["batch_size"])

data_name = prm_structure["corpus_name" ] + '_valid'
valid_mb_set_x,valid_mb_set_y,valid_mb_set_m = load_minibatches(prm_structure["data_location"], data_name, prm_structure["batch_size"])

n_in = train_mb_set_x[0].shape[2]
n_out = train_mb_set_y[0].shape[2]

print "# Loading duration: ",time.time()-time_0 ," sec"


#### Hyper parameter
prm_structure["net_size"      ] = [n_in,10, n_out]
prm_structure["hidden_layer"  ] = prm_structure["net_size"].__len__() - 2
prm_structure["bi_directional"] = False
prm_structure["identity_func" ] = False
prm_structure["train_set_len" ] = train_mb_set_x.__len__()
prm_structure["valid_set_len" ] = valid_mb_set_x.__len__()
prm_structure["output_location"] = "outcome/"
prm_structure["output_type"    ] = "both"        # console, file, both

prm_optimization["epochs"        ] = 5
prm_optimization["learn_rate"    ] = 0.0001
prm_optimization["lr_decline"    ] = 0.95
prm_optimization["momentum"      ] = 0.9
prm_optimization["decay_rate"    ] = 0.9
prm_optimization["use_dropout"   ] = False       # False, True
prm_optimization["dropout_level" ] = 0.5
prm_optimization["regularization"] = False       # False, L2, ( L1 )
prm_optimization["reg_factor"    ] = 0.01
prm_optimization["optimization"  ] = "adadelta"  # sgd, nm_rmsprop, rmsprop, nesterov_momentum, adadelta
prm_optimization["noisy_input"   ] = False       # False, True
prm_optimization["noise_level"   ] = 0.6
prm_optimization["loss_function" ] = "cross_entropy" # w2_cross_entropy, cross_entropy
prm_optimization["bound_weight"  ] = False       # False, Integer (2,12)


###### Build model

lstm = rnnModel(prm_structure, prm_optimization, rng, trng)

lstm.print_model_params()
lstm.pub("# Build model")

time_1 = time.time()
lstm.pub("Model build time"+ str(time_1-time_0) + "sec")

train_fn    = lstm.get_training_function()
valid_fn    = lstm.get_validation_function()
forward_fn  = lstm.get_forward_function()


###### START TRAINING
lstm.pub("Start training")

batch_order = np.arange(0,prm_structure["train_set_len"])

for i in xrange(prm_optimization["epochs"]):
    time_training_start = time.time()
    time_training_iteration = time_training_start
    lstm.pub("------------------------------------------")
    lstm.pub(str(i)+" Epoch, Training run")


    train_error = np.zeros(prm_structure["train_set_len" ])
    batch_permut = rng.permutation(batch_order)

    for j in batch_order:

        train_error[j], net_out = train_fn( train_mb_set_x[batch_permut[j]],
                                            train_mb_set_y[batch_permut[j]],
                                            train_mb_set_m[batch_permut[j]]
                                            )

        #Insample error
        if ( j%50) == 0 :
            lstm.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_training_iteration) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_training_iteration = time.time()

        #Validation
        if ( (j%500) == 0 or j == prm_structure["train_set_len" ]-1 ) and j>0:
            lstm.pub("###########################################")
            lstm.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([prm_structure["valid_set_len"]])
            ce_error = np.zeros([prm_structure["valid_set_len"]*prm_structure["batch_size"]])
            auc_error = np.zeros([prm_structure["valid_set_len"]*prm_structure["batch_size"]])

            for v in np.arange(0,prm_structure["valid_set_len"]):
                v_net_out_, v_error[v] = valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])

                for b in np.arange(0,prm_structure["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]

                    count = v * prm_structure["batch_size"] + b

                    ce_error[count] = sklearn.metrics.log_loss( true_out,code_out)
                    auc_error[count] = sklearn.metrics.roc_auc_score( true_out,code_out)

            lstm.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error)))
            lstm.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
            lstm.pub("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error)))
            lstm.pub("###########################################")

            lstm.dump() #save current weights

    lstm.pub("###########################################")
    lstm.pub("Insample Error: " + str(np.mean(train_error)))
    lstm.pub("Epoch training duration: "+ str(time.time()-time_training_start) + "sec")

#Finale
lstm.pub("## ||||||||||||||||||||||||||||||||||||||||")


