#!/usr/bin/env python
__author__ = 'joerg'

""" Little timer task """
"""___________________"""
"""
"""


######  GLOBAL THEANO CONFIG   #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False" #ast_run
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags


######         IMPORTS          ######
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import sklearn.metrics
from scipy import stats
import time
from collections import OrderedDict

from rnnfwk.build_model import rnnModel
from rnnfwk.data_handler import load_minibatches



###### GLOBAL TIMER
time_0 = time.time()

########## RANDOM STREAMS
params_optimization = OrderedDict()
params_optimization["seed"] = 211
rng = np.random.RandomState(params_optimization["seed"])
trng = RandomStreams(params_optimization["seed"] )




###### DATA IN
print("# Load data")
params_structure = OrderedDict()
params_structure["batch_size"    ] = 10
params_structure["set_specs" ] = ""
params_structure["corpus_name" ] = "little-timer"

data_location = "data_set/"

data_name = params_structure["corpus_name" ] + '_train' + params_structure["set_specs" ]
train_mb_set_x,train_mb_set_y,train_mb_set_m = load_minibatches(data_location, data_name, params_structure["batch_size"])

data_name = params_structure["corpus_name" ] + '_valid' + params_structure["set_specs" ]
valid_mb_set_x,valid_mb_set_y,valid_mb_set_m = load_minibatches(data_location, data_name, params_structure["batch_size"])



input_size = train_mb_set_x[0].shape[2]
output_size = train_mb_set_y[0].shape[2]


print "# Loading duration: ",time.time()-time_0 ," sec"


#### Hyper parameter

params_structure["net_size"      ] = [input_size,10, output_size]
params_structure["hidden_layer"  ] = params_structure["net_size"].__len__() - 2
params_structure["bi_directional"] = True
params_structure["identity_func" ] = False
params_structure["train_set_len" ] = train_mb_set_x.__len__()
params_structure["valid_set_len" ] = valid_mb_set_x.__len__()
params_structure["output_location"] = "outcome/"
params_structure["output_type"    ] = "both"        # console, file, both


params_optimization["epochs"        ] = 2
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
params_optimization["bound_weight"  ] = False       # False, Integer (2,12)


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

            v_error = np.zeros([params_structure["valid_set_len"]])
            ce_error = np.zeros([params_structure["valid_set_len"]*params_structure["batch_size"]])
            auc_error = np.zeros([params_structure["valid_set_len"]*params_structure["batch_size"]])

            for v in np.arange(0,params_structure["valid_set_len"]):
                v_net_out_, v_error[v] = valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])

                for b in np.arange(0,params_structure["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]

                    count = v * params_structure["batch_size"] + b

                    ce_error[count] = sklearn.metrics.log_loss( true_out,code_out)
                    auc_error[count] = sklearn.metrics.roc_auc_score( true_out,code_out)



            lstm.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error))) #str(np.mean(v_error)))
            lstm.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error))) #str(np.mean(ce_error)))
            lstm.pub("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error))) #str(np.mean(acc_error)))


            #lstm.pub("## area under curve 10ms : " + "{0:.4f}".format(area_under_curve))
            lstm.pub("###########################################")

            #lstm.dump()



    #t_learn_rate = t_learn_rate * params_optimization['lr_decline']
    lstm.pub("###########################################")
    lstm.pub("Insample Error: " + str(np.mean(train_error)))
    #lstm.pub("Lerning Rate adaptation: " + str(t_learn_rate.eval()) )
    lstm.pub("Epoch training duration: "+ str(time.time()-time_training_start) + "sec")

#Finale Test
lstm.pub("## ||||||||||||||||||||||||||||||||||||||||")

lstm.dump()

import matplotlib.pyplot as plt
data_name = params_structure["corpus_name" ] + '_' + 'test' + params_structure["set_specs" ]
test_mb_set_x,test_mb_set_y,test_mb_set_m = load_minibatches(data_location, data_name, params_structure["batch_size"])
set_length = test_mb_set_x.__len__()

###### LOAD MODEL
#model_name = "outcome/" + "n-2-10-2_d-13-08-2016_v-10.prm"
#lstm = rnnModel(None, None, rng, trng, True, model_name)

forward_fn  = lstm.get_forward_function()

###### TEST MODEL
ce_error = np.zeros([set_length*params_structure["batch_size"]])
auc_error = np.zeros([set_length*params_structure["batch_size"]])

for v in np.arange(0,set_length):
    v_net_out_ = forward_fn(test_mb_set_x[v], test_mb_set_m[v])[0]

    for b in np.arange(0,params_structure["batch_size"]):
        true_out = test_mb_set_y[v][:,b,:]
        code_out = v_net_out_[:,b,:]

        count = v * params_structure["batch_size"] + b

        ce_error[count] = sklearn.metrics.log_loss( true_out,code_out)
        auc_error[count] = sklearn.metrics.roc_auc_score( true_out,code_out)


print("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error))) #str(np.mean(ce_error)))
print("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error))) #str(np.mean(acc_error)))


###### PLOT SAMPLE

sample_no = 0
batch = 0
net_out = forward_fn(test_mb_set_x[sample_no], test_mb_set_m[sample_no])[0]

fig = plt.figure()
fig.suptitle('Little timer task - Sample')
plt.subplot(2,2,1)
plt.xlabel('start signal')
plt.plot(test_mb_set_x[sample_no][:,batch,0])
plt.ylim([0,1.1])
plt.xlim([0,80])
plt.subplot(2,2,3)
plt.xlabel('duration signal')
plt.plot(test_mb_set_x[sample_no][:,batch,1])
plt.ylim([0,9.1])
plt.xlim([0,80])
plt.subplot(1,2,2)
plt.xlabel('target signal')
plt.plot(test_mb_set_y[sample_no][:,batch,0])
plt.plot(net_out[:,batch,0])
plt.ylim([0,1.1])
plt.xlim([0,80])
plt.show()