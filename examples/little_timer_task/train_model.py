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
import numpy as np
import sklearn.metrics
import time
from collections import OrderedDict
from recnet.build_model import rnnModel


### 1. Step: Define parameters
parameter = OrderedDict()
parameter["train_data_name"] = "little-timer_train.klepto"
parameter["valid_data_name"] = "little-timer_valid.klepto"
parameter["data_location"] = "data_set/"
parameter["batch_size" ] = 10

parameter["net_size"      ] = [      2,     10,         2]
parameter["net_unit_type" ] = ['input', 'LSTM', 'softmax']
parameter["net_act_type"  ] = [    '-',  'tanh',      '-']
parameter["net_arch"      ] = [    '-',    'bi',     'ff']

parameter["epochs"        ] = 5
parameter["learn_rate"    ] = 0.0001
parameter["use_dropout"   ] = False       # False, True
parameter["regularization"] = False       # False, L2,  L1
parameter["optimization"  ] = "adadelta"  # sgd, nm_rmsprop, rmsprop, nesterov_momentum, adadelta
parameter["noisy_input"   ] = False       # False, True
parameter["loss_function" ] = "cross_entropy" # w2_cross_entropy, cross_entropy



### 2. Step: Create new model
model = rnnModel(parameter)


### 3. Step: Build model functions
train_fn    = model.get_training_function()
valid_fn    = model.get_validation_function()


### 4. Step: Train model

### 4.1: Create minibatches

### 4.2: Train model with minibatch

### 4.3: Plot insample error during training

### 4.4: Plot validation error during training

###### GLOBAL TIMER
time_0 = time.time()










###### START TRAINING
model.pub("Start training")




model.mbh.create_mini_batches("valid")
valid_mb_set_x, valid_mb_set_y, valid_mb_set_m = model.mbh.load_mini_batches("valid")

for i in xrange(model.prm.optimize["epochs"]):
    time_training_start = time.time()
    time_training_iteration = time_training_start
    model.pub("------------------------------------------")
    model.pub(str(i)+" Epoch, Training run")


    model.mbh.create_mini_batches("train")
    mb_train_x, mb_train_y, mb_mask = model.mbh.load_mini_batches("train")


    train_error = np.zeros(model.prm.data["train_set_len" ])


    for j in xrange(model.prm.data["train_batch_quantity"]):




        net_out, train_error[j] = train_fn( mb_train_x[j],
                                            mb_train_y[j],
                                            mb_mask[j]
                                            )

        #Insample error
        if ( j%50) == 0 :
            model.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_training_iteration) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_training_iteration = time.time()

        #Validation
        if ( (j%500) == 0 or j == model.prm.data["train_set_len" ]-1 ) and j>0:
            model.pub("###########################################")
            model.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([model.prm.data["valid_batch_quantity"]])
            ce_error = np.zeros([model.prm.data["valid_batch_quantity"]*model.prm.data["batch_size"]])
            auc_error = np.zeros([model.prm.data["valid_batch_quantity"]*model.prm.data["batch_size"]])

            for v in np.arange(0,model.prm.data["valid_batch_quantity"]):
                v_net_out_, v_error[v] = valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])

                for b in np.arange(0,model.prm.data["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]

                    count = v * model.prm.data["batch_size"] + b

                    ce_error[count] = sklearn.metrics.log_loss( true_out,code_out)
                    auc_error[count] = sklearn.metrics.roc_auc_score( true_out,code_out)

            model.pub("## cross entropy theano  : " + "{0:.4f}".format(np.mean(v_error)))
            model.pub("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
            model.pub("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error)))
            model.pub("###########################################")

            model.dump() #save current weights


    model.pub("###########################################")
    model.pub("Insample Error: " + str(np.mean(train_error)))
    model.pub("Epoch training duration: "+ str(time.time()-time_training_start) + "sec")

#Finale
model.pub("## ||||||||||||||||||||||||||||||||||||||||")


