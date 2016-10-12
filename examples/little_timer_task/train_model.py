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
import klepto

from recnet.build_model import rnnModel
from recnet.data_handler import load_minibatches


### 1. Step: Create new model

### 2. Step: Pass data to model

train_x
train_y

valid_x
valid_y



### 3. Step: Define parameters

### 4. Step: Build model functions

### 5. Step: Train model

### 5.1: Create minibatches

### 5.2: Train model with minibatch

### 5.3: Plot insample error during training

### 5.4: Plot validation error during training

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
prm_structure["net_unit_type" ] = ['input', 'tanh', 'softmax']
prm_structure["hidden_layer"  ] = prm_structure["net_size"].__len__() - 2
prm_structure["bi_directional"] = False
prm_structure["identity_func" ] = False
prm_structure["train_set_len" ] = train_mb_set_x.__len__()
prm_structure["valid_set_len" ] = valid_mb_set_x.__len__()
if "log" not in os.listdir(os.getcwd()):
    os.mkdir("log")
prm_structure["output_location"] = "log/"
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


################################ NEW DATA HANDLING
file_name = "data_set/little-timer_train.klepto"
d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
d.load()
train_data_x = d['x']
train_data_y = d['y']
d.clear()


train_data_x_len = [i.__len__() for i in train_data_x]
train_data_y_len = [i.__len__() for i in train_data_y]
assert np.array_equal(train_data_x_len, train_data_y_len), "x and y sequences have not the same length"

prm_structure["x_length"] = train_data_x[0].shape[1]
prm_structure["y_length"] = train_data_y[0].shape[1]

lstm.p_struct["batch_size"] = 10
lstm.p_struct["train_set_len" ] = train_data_x.__len__()
lstm.p_struct["batch_quantity"] = int(np.trunc(lstm.p_struct["train_set_len" ]/lstm.p_struct["batch_size"]))


sample_order = np.arange(0,lstm.p_struct["train_set_len" ])
sample_order = rng.permutation(sample_order)

##################################################
batch_size = lstm.p_struct["batch_size"]
x_length = train_data_x[0].shape[1] #prm_structure["x_length"]
y_length = train_data_y[0].shape[1] # prm_structure["y_length"]
##################################################

for i in xrange(prm_optimization["epochs"]):
    time_training_start = time.time()
    time_training_iteration = time_training_start
    lstm.pub("------------------------------------------")
    lstm.pub(str(i)+" Epoch, Training run")


    train_error = np.zeros(prm_structure["train_set_len" ])
    batch_permut = rng.permutation(batch_order)


    for j in xrange(lstm.p_struct["batch_quantity"]):

        ### build minibatch
        sample_selection = sample_order[ j*batch_size:j*batch_size+batch_size ]
        max_seq_len = np.max(  [train_data_x[i].__len__() for i in sample_selection])

        mb_train_x = np.zeros([max_seq_len, batch_size, x_length])
        mb_train_y = np.zeros([max_seq_len, batch_size, y_length])
        mb_mask = np.zeros([max_seq_len, batch_size, 1])

        for k in xrange(batch_size):
            s = sample_selection[k]
            sample_length =  train_data_x_len[s]
            mb_train_x[:sample_length,k,:] = train_data_x[s][:sample_length]
            mb_train_y[:sample_length,k,:] = train_data_y[s][:sample_length]
            mb_mask[:sample_length,k,:] = np.ones([sample_length,1])

        ### build minibatch end


        net_out, train_error[j] = train_fn( mb_train_x,
                                            mb_train_y,
                                            mb_mask
                                            )

    # for j in batch_order:
    #     net_out, train_error[j] = train_fn( train_mb_set_x[batch_permut[j]],
    #                                         train_mb_set_y[batch_permut[j]],
    #                                         train_mb_set_m[batch_permut[j]]
    #                                         )

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


