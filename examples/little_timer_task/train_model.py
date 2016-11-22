#!/usr/bin/env python

""" Little timer task """
"""___________________"""
"""    TRAIN MODEL
"""


######  Set global Theano config  #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False"
print("Theano Flags: " + t_flags)
os.environ["THEANO_FLAGS"] = t_flags


######         Imports          ######
import numpy as np
import sklearn.metrics
import time
from past.builtins import xrange
import recnet


### 1. Step: Create new model
rn = recnet.rnnModel()

### 2. Step: Define parameters
rn.parameter["train_data_name"] = "little-timer_train.klepto"
rn.parameter["valid_data_name"] = "little-timer_valid.klepto"
rn.parameter["data_location"  ] = "data_set/"
rn.parameter["batch_size"     ] = 10

rn.parameter["net_size"       ] = [      2,     10,         2]
rn.parameter["net_unit_type"  ] = ['input',  'LSTMp_ln', 'softmax']
rn.parameter["net_act_type"   ] = [    '-',  'tanh',      '-']
rn.parameter["net_arch"       ] = [    '-',    'bi',     'ff']

rn.parameter["epochs"         ] = 5
rn.parameter["use_dropout"    ] = False
rn.parameter["regularization" ] = 'L2'
rn.parameter["reg_factor"     ] = 0.1
rn.parameter["optimization"   ] = "adadelta"
rn.parameter["loss_function"  ] = "cross_entropy"

### 3. Step: Create model and compile functions
rn.create(['train', 'valid'])

### 4. Step: Train model
rn.pub("Start training")

### 4.1: Create minibatches for validation set
mb_valid_x, mb_valid_y, mb_valid_m = rn.get_mini_batches("valid")

### 4.2: Start epoch loop
for i in range(rn.epochs()):
    rn.pub("------------------------------------------")
    rn.pub(str(i+1)+" Epoch, Training run")
    time_0 = time.time()
    time_1 = time.time()

    ### 4.3: Create minibatches for training set
    mb_train_x, mb_train_y, mb_train_m = rn.get_mini_batches("train")

    ### 4.4: Iterate over mini batches
    train_error = np.zeros(rn.sample_quantity('train'))
    for j in xrange(rn.batch_quantity('train')):

        ### 4.5: Train with one mini batch
        net_out, train_error[j] = rn.train_fn( mb_train_x[j], mb_train_y[j], mb_train_m[j])

        ### 4.6: Print training error
        if ( j%50) == 0 :
            rn.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_0) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_0 = time.time()

        ### 4.7: Print validation error
        if ( (j%500) == 0 or j == rn.batch_quantity('train')-1 ) and j>0:
            rn.pub("###########################################")
            rn.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([rn.batch_quantity('valid')])
            ce_error = np.zeros([rn.sample_quantity('valid')])
            auc_error = np.zeros([rn.sample_quantity('valid')])

            for v in np.arange(0,rn.batch_quantity('valid')):
                v_net_out_, v_error[v] = rn.valid_fn(mb_valid_x[v],mb_valid_y[v],mb_valid_m[v])

                for b in np.arange(0,rn.batch_size()):
                    true_out = mb_valid_y[v][:, b, :]
                    code_out = v_net_out_[:, b, :]

                    count = v * rn.batch_size() + b

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
    rn.pub("Epoch training duration: " + str(time.time()-time_1) + "sec")

#Finish training
rn.pub("## ||||||||||||||||||||||||||||||||||||||||")