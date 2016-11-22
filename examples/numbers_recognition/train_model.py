#!/usr/bin/env python

""" Numbers recognition """
"""____________________"""
"""     TRAIN MODEL
"""


######  Set global Theano config  #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False"
print("Theano Flags: " + t_flags)
os.environ["THEANO_FLAGS"] = t_flags


######         Imports          ######
import numpy as np
import time
from past.builtins import xrange
import recnet
from .util import edit_distance

### 1. Step: Create new model
rn = recnet.rnnModel()

### 2. Step: Define parameters
rn.parameter["train_data_name"] = "numbers_image_train.klepto"
rn.parameter["valid_data_name"] = "numbers_image_valid.klepto"
rn.parameter["data_location"  ] = "data_set/"
rn.parameter["batch_size"     ] = 1

rn.parameter["net_size"       ] = [      9,      10,      10+1]
rn.parameter["net_unit_type"  ] = ['input',  'conv', 'softmax']
rn.parameter["net_act_type"   ] = [    '-',  'tanh',       '-']
rn.parameter["net_arch"       ] = [    '-',    'bi',      'ff']

rn.parameter["random_seed"    ] = 211
rn.parameter["epochs"         ] = 30
rn.parameter["learn_rate"     ] = 0.001
rn.parameter["regularization" ] = 'L2'
rn.parameter["reg_factor"     ] = 0.1
rn.parameter["optimization"   ] = "sgd"
rn.parameter["loss_function"  ] = "CTClog"

### 3. Step: Create model and compile functions
rn.create(['train', 'valid'])

### 4. Step: Train model
rn.pub("Start training")

### 4.1: Create minibatches for training/validation set
mb_train_x, mb_train_y, mb_train_m = rn.get_mini_batches("train")
mb_valid_x, mb_valid_y, mb_valid_m = rn.get_mini_batches("valid")

### 4.2: Start epoch loop
for i in xrange(rn.epochs()):
    time_0 = time.time()

    ### 4.3: Iterate over mini batches and train model
    train_error = np.zeros(rn.sample_quantity('train'))
    for j in xrange(rn.batch_quantity('train')):
        net_out, train_error[j] = rn.train_fn( mb_train_x[j], mb_train_y[j], mb_train_m[j])

    rn.pub(str(i)+" Epoch, Mean Insample Error: " + "{0:.4f}".format(np.mean(train_error)) + \
           ", Duration: "+ "{0:.4f}".format(time.time()-time_0) + "sec" )

    ### 4.4: Validation, calculate the mean edit distance
    if(i % 5 == 0 and i != 0) or i == (rn.epochs() - 1):

        valid_error = np.zeros([rn.batch_quantity('valid'),rn.batch_size()])
        time_0 = time.time()
        for j in xrange(rn.batch_quantity('valid')):
            net_out, v_error = rn.valid_fn( mb_valid_x[j], mb_valid_y[j], mb_valid_m[j])

            for b in xrange(rn.batch_size()):
                true_out = mb_valid_y[j][b,:]
                cln_true_out = np.delete(true_out, np.where(true_out == 10))

                net_out = net_out[:,b,:]
                arg_net_out = np.argmax(net_out, axis=1)
                cln_net_out = np.delete(arg_net_out, np.where(arg_net_out == 10))

                valid_error[j,b] =   edit_distance(cln_true_out, cln_net_out)

        rn.pub("Validation, Mean Edit Distance: " + "{0:.4f}".format(np.mean(valid_error)) + \
           ", Duration: "+ "{0:.4f}".format(time.time()-time_0) + "sec" )

        ### 4.5: Make a dump of the current model
        rn.dump()
#Finish training
rn.pub("## ||||||||||||||||||||||||||||||||||||||||")
