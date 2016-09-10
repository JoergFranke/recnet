#!/usr/bin/env python

""" Little timer task """
"""___________________"""
"""    USE MODEL
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
from collections import OrderedDict
import matplotlib.pyplot as plt

from recnet.build_model import rnnModel
from recnet.data_handler import load_minibatches


########## RANDOM STREAMS
prm_optimization = OrderedDict()
prm_optimization["seed"] = 211
rng = np.random.RandomState(prm_optimization["seed"])
trng = RandomStreams(prm_optimization["seed"] )


###### DATA IN
print("# Load data")
prm_structure = OrderedDict()
prm_structure["batch_size"  ] = 10
prm_structure["corpus_name" ] = "little-timer"
prm_structure["data_location"] = "data_set/"

data_name = prm_structure["corpus_name" ] + '_' + 'test'
test_mb_set_x,test_mb_set_y,test_mb_set_m = load_minibatches(prm_structure["data_location"], data_name, prm_structure["batch_size"])
set_length = test_mb_set_x.__len__()


###### LOAD MODEL
################################################################################
###################### ADD NAME FROM TRAINED MODEL HERE ! ######################
model_name = "outcome/" + "n-*********************.prm"
lstm = rnnModel(None, None, rng, trng, True, model_name, 10)

forward_fn = lstm.get_forward_function()


###### TEST MODEL
ce_error = np.zeros([set_length*prm_structure["batch_size"]])
auc_error = np.zeros([set_length*prm_structure["batch_size"]])

for v in np.arange(0, set_length):
    v_net_out_ = forward_fn(test_mb_set_x[v], test_mb_set_m[v])[0]

    for b in np.arange(0,prm_structure["batch_size"]):
        true_out = test_mb_set_y[v][:, b, :]
        code_out = v_net_out_[:, b, :]

        count = v * prm_structure["batch_size"] + b

        ce_error[count] = sklearn.metrics.log_loss(true_out, code_out)
        auc_error[count] = sklearn.metrics.roc_auc_score(true_out, code_out)


print("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
print("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error)))


###### PLOT SAMPLE
sample_no = 1
batch = 1
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
plt.plot(test_mb_set_y[sample_no][:,batch,0], label="Target")
plt.plot(net_out[:,batch,0], label="LSTM Output")
plt.legend(loc='upper right',frameon=True)
plt.ylim([0,1.1])
plt.xlim([0,80])
plt.show()