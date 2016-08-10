#!/usr/bin/env python
__author__ = 'joerg'

""" Phonem Classification on the TIMIT speech corpus with Theano based DBLSTM Network """
"""___________________________________________________________________________________"""
"""
"""


######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False" #ast_run
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags

######      THEANO CONFIG      #######
import theano
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

######         IMPORTS          ######
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import sklearn.metrics
from scipy import stats
import time
from collections import OrderedDict

from rnnfwk.build_model import rnnModel
from rnnfwk.data_handler import load_minibatches

from evaluation_functions import boundary_evaluation

validation_batches = boundary_evaluation.validation_batches

###### GLOBAL TIMER
time_0 = time.time()

########## RANDOM STREAMS
params_optimization = OrderedDict()
params_optimization["seed"] = 211
rng = np.random.RandomState(params_optimization["seed"])
rng.seed(params_optimization["seed"])
trng = RandomStreams(params_optimization["seed"] )
trng.seed(params_optimization["seed"])



###### DATA IN
print("# Load data")
params_structure = OrderedDict()
params_structure["batch_size"    ] = 10
params_structure["set_specs" ] = "xy_mfcc12-26win25-10"
params_structure["corpus_name" ] = "timit"

data_location = "data_set/"

data_name = params_structure["corpus_name" ] + '_train_' + params_structure["set_specs" ]
train_mb_set_x,train_mb_set_y,train_mb_set_m = load_minibatches(data_location, data_name, params_structure["batch_size"])

data_name = params_structure["corpus_name" ] + '_valid_' + params_structure["set_specs" ]
valid_mb_set_x,valid_mb_set_y,valid_mb_set_m = load_minibatches(data_location, data_name, params_structure["batch_size"])



input_size = train_mb_set_x[0].shape[2]
output_size = train_mb_set_y[0].shape[2]


print "# Loading duration: ",time.time()-time_0 ," sec"


#### Hyper parameter

params_structure["net_size"      ] = [input_size,100, output_size]
params_structure["hidden_layer"  ] = params_structure["net_size"].__len__() - 2
params_structure["bi_directional"] = True
params_structure["identity_func" ] = False
params_structure["train_set_len" ] = train_mb_set_x.__len__()
params_structure["valid_set_len" ] = valid_mb_set_x.__len__()
params_structure["output_location"] = "outcome/"
params_structure["output_type"    ] = "both"        # console, file, both


params_optimization["epochs"        ] = 20
params_optimization["learn_rate"    ] = 0.0001
params_optimization["lr_decline"    ] = 0.95
params_optimization["momentum"      ] = 0.9
params_optimization["decay_rate"    ] = 0.9
params_optimization["use_dropout"   ] = False       # False, True
params_optimization["dropout_level" ] = 0.5
params_optimization["regularization"] = False       # False, L2, ( L1 )
params_optimization["reg_factor"    ] = 0.01
params_optimization["optimization"  ] = "adadelta"  # sgd, nm_rmsprop, rmsprop, nesterov_momentum, adadelta
params_optimization["noisy_input"   ] = False       # False, True
params_optimization["noise_level"   ] = 0.6
params_optimization["loss_function" ] = "cross_entropy" # w2_cross_entropy, cross_entropy
params_optimization["bound_weight"  ] = 3       # False, Integer (2,12)


###### Build model

lstm = rnnModel(params_structure, params_optimization, rng, trng)

lstm.print_model_params()
lstm.pub("# Build model")


time_1 = time.time()
lstm.pub("Model build time"+ str(time_1-time_0) + "sec")

train_fn    = lstm.get_training_function()
valid_fn    = lstm.get_validation_function()
forward_fn  = lstm.get_forward_function()

###### START TRAINING
lstm.pub("Start training")

batch_order = np.arange(0,params_structure["train_set_len"])

#save measurements
list_acc = []
list_f1s = []
list_tce = []



for i in xrange(params_optimization["epochs"]):
    time_training_start = time.time()
    time_training_temp = time_training_start
    lstm.pub("------------------------------------------")
    lstm.pub(str(i)+" Epoch, Training run")


    train_error = np.zeros(params_structure["train_set_len" ])
    batch_permut = rng.permutation(batch_order)

    for j in batch_order:

        train_error[j], net_out = train_fn( train_mb_set_x[batch_permut[j]],
                                            train_mb_set_y[batch_permut[j]],
                                            train_mb_set_m[batch_permut[j]]
                                            )

        #save training error in class
        #print(net_out[:,1])
        #print( train_mb_set_m[batch_permut[j]][:,1])

        #Insample error
        if ( j%50) == 0 :
            lstm.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_training_temp) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_training_temp = time.time()

        #Validation
        if ( (j%500) == 0 or j == params_structure["train_set_len" ]-1 ) and j>0:
            lstm.pub("###########################################")
            lstm.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_net_out = []
            v_error = np.zeros([params_structure["valid_set_len"]])

            v_error = np.zeros([params_structure["valid_set_len"]])
            ce_error = np.zeros([params_structure["valid_set_len"],params_structure["batch_size"]])
            acc_error = np.zeros([params_structure["valid_set_len"],params_structure["batch_size"]])

            for v in np.arange(0,params_structure["valid_set_len"]):
                v_net_out_, v_error[v] = valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])
                v_net_out.append(v_net_out_)

                for b in np.arange(0,params_structure["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]
                    ce_error[v,b] = sklearn.metrics.log_loss( true_out,code_out)
                    #acc_error[v,b] = np.mean(np.equal(np.argmax(true_out,axis=1), np.argmax(code_out,axis=1)))
                    acc_error[v,b] = np.mean(np.argmax(true_out,axis=1) * v_net_out_[:,b,1])
                    # todo correct acc for boundaries problem

            accuracy,f1_score,area_under_curve = validation_batches(v_net_out, valid_mb_set_y,valid_mb_set_m, 2)

            list_acc.append(np.mean(v_error))
            list_tce.append(f1_score[0])
            array_acc = np.asarray(list_acc[-3:])
            array_tce = np.asarray(list_tce[-3:])
            acc_slope, intercept, r_value, p_value, std_err = stats.linregress(range(array_acc.shape[0]),array_acc)
            tce_slope, intercept, r_value, p_value, std_err = stats.linregress(range(array_tce.shape[0]),array_tce)

            lstm.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error))) #str(np.mean(v_error)))
            lstm.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error))) #str(np.mean(ce_error)))
            lstm.pub("## correct classified    : " + "{0:.4f}".format(np.mean(acc_error))) #str(np.mean(acc_error)))
            lstm.pub("## accuracy 10ms         : " + "{0:.4f}".format(accuracy[0]))
            lstm.pub("## f1 score 10ms         : " + "{0:.4f}".format(f1_score[0]))

            lstm.pub("## cetheano improve      : " + "{0:.6f}".format(acc_slope))
            lstm.pub("## f1 score improve      : " + "{0:.6f}".format(tce_slope))

            #lstm.pub("## area under curve 10ms : " + "{0:.4f}".format(area_under_curve))
            lstm.pub("###########################################")

            lstm.dump()



    #t_learn_rate = t_learn_rate * params_optimization['lr_decline']
    lstm.pub("###########################################")
    lstm.pub("Insample Error: " + str(np.mean(train_error)))
    #lstm.pub("Lerning Rate adaptation: " + str(t_learn_rate.eval()) )
    lstm.pub("Epoch training duration: "+ str(time.time()-time_training_start) + "sec")

#Finale Test
lstm.pub("## ||||||||||||||||||||||||||||||||||||||||")


