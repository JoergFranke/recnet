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
model = rnnModel()





### 2. Step: Define parameters
parameter = OrderedDict()
parameter["minibatch_location"] = "minibatch"
parameter["output_location"] = "log"
parameter["output_type"    ] = "both"        # console, file, both

parameter["train_data_name"] = "little-timer_train.klepto"
parameter["valid_data_name"] = "little-timer_valid.klepto"
parameter["data_location"] = "data_set/"

parameter["batch_size" ] = 10 #todo is data no structure
parameter["net_size"      ] = [2, 10, 2]
parameter["net_unit_type" ] = ['input', 'LSTM', 'softmax']
parameter["net_act_type" ] = ['-', 'tanh', '-']
parameter["bi_directional"] = False

parameter["random_seed"   ] = 211
parameter["epochs"        ] = 5
parameter["learn_rate"    ] = 0.0001
parameter["momentum_rate" ] = 0.9
parameter["decay_rate"    ] = 0.9
parameter["use_dropout"   ] = False       # False, True
parameter["dropout_level" ] = 0.5
parameter["regularization"] = False       # False, L2, ( L1 )
parameter["reg_factor"    ] = 0.01
parameter["optimization"  ] = "adadelta"  # sgd, nm_rmsprop, rmsprop, nesterov_momentum, adadelta
parameter["noisy_input"   ] = False       # False, True
parameter["noise_level"   ] = 0.6
parameter["loss_function" ] = "cross_entropy" # w2_cross_entropy, cross_entropy
parameter["bound_weight"  ] = False       # False, Integer (2,12)

model.pass_parameter_dict(parameter)



### 3. Step: Pass data to model

# train_x
# train_y
#
# valid_x
# valid_y


# update ->
# parameter["train_set_len" ] = train_mb_set_x.__len__()
# parameter["valid_set_len" ] = valid_mb_set_x.__len__()





### 4. Step: Build model functions
model.build_model()
### 5. Step: Train model

### 5.1: Create minibatches

### 5.2: Train model with minibatch

### 5.3: Plot insample error during training

### 5.4: Plot validation error during training

###### GLOBAL TIMER
time_0 = time.time()


########## RANDOM STREAMS





###### DATA IN
print("# Load data")


data_name = "little-timer_train"
train_mb_set_x,train_mb_set_y,train_mb_set_m = load_minibatches(parameter["data_location"], data_name, parameter["batch_size"])

data_name = "little-timer_valid"
valid_mb_set_x,valid_mb_set_y,valid_mb_set_m = load_minibatches(parameter["data_location"], data_name, parameter["batch_size"])

n_in = train_mb_set_x[0].shape[2]
n_out = train_mb_set_y[0].shape[2]



model.prm.data["train_set_len" ] = train_mb_set_x.__len__()
model.prm.data["valid_set_len" ] = valid_mb_set_x.__len__()

print "# Loading duration: ",time.time()-time_0 ," sec"


#### Hyper parameter





###### Build model



model.print_model_params()
model.pub("# Build model")

time_1 = time.time()
model.pub("Model build time"+ str(time_1-time_0) + "sec")

train_fn    = model.get_training_function()
valid_fn    = model.get_validation_function()
forward_fn  = model.get_forward_function()


###### START TRAINING
model.pub("Start training")

batch_order = np.arange(0,model.prm.data["train_set_len"])


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

model.prm.struct["x_length"] = train_data_x[0].shape[1]
model.prm.struct["y_length"] = train_data_y[0].shape[1]

model.prm.struct["batch_size"] = 10
model.prm.data["train_set_len" ] = train_data_x.__len__()
model.prm.struct["batch_quantity"] = int(np.trunc(model.prm.struct["train_set_len" ]/model.prm.struct["batch_size"]))


sample_order = np.arange(0,model.prm.data["train_set_len" ])
sample_order = model.rng.permutation(sample_order)

##################################################
batch_size = model.prm.struct["batch_size"]
x_length = train_data_x[0].shape[1] #model.prm.struct["x_length"]
y_length = train_data_y[0].shape[1] # model.prm.struct["y_length"]
##################################################

for i in xrange(model.prm.optimize["epochs"]):
    time_training_start = time.time()
    time_training_iteration = time_training_start
    model.pub("------------------------------------------")
    model.pub(str(i)+" Epoch, Training run")


    train_error = np.zeros(model.prm.data["train_set_len" ])
    batch_permut = model.rng.permutation(batch_order)


    for j in xrange(model.prm.struct["batch_quantity"]):

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
            model.pub("counter: " + "{:3.0f}".format(j)
                   + "  time: " + "{:5.2f}".format(time.time()-time_training_iteration) + "sec"
                   + "  error: " + "{:6.4f}".format(train_error[j]))
            time_training_iteration = time.time()

        #Validation
        if ( (j%500) == 0 or j == model.prm.data["train_set_len" ]-1 ) and j>0:
            model.pub("###########################################")
            model.pub("## epoch validation at " + str(i) + "/" + str(j))

            v_error = np.zeros([model.prm.data["valid_set_len"]])
            ce_error = np.zeros([model.prm.data["valid_set_len"]*model.prm.struct["batch_size"]])
            auc_error = np.zeros([model.prm.data["valid_set_len"]*model.prm.struct["batch_size"]])

            for v in np.arange(0,model.prm.data["valid_set_len"]):
                v_net_out_, v_error[v] = valid_fn(valid_mb_set_x[v],valid_mb_set_y[v],valid_mb_set_m[v])

                for b in np.arange(0,model.prm.struct["batch_size"]):
                    true_out = valid_mb_set_y[v][:,b,:]
                    code_out = v_net_out_[:,b,:]

                    count = v * model.prm.struct["batch_size"] + b

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


