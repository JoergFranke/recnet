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
import numpy as np
import sklearn.metrics
from collections import OrderedDict
import matplotlib.pyplot as plt

from recnet.build_model import rnnModel


### 1. Step: Define parameters
parameter = OrderedDict()
parameter["load_model"] = True
parameter["model_location"] = "model_save/"

###### LOAD MODEL
################################################################################
########################### ADD NAME FROM TRAINED MODEL HERE ! #################
parameter["model_name"] = "**********************************.prm"
parameter["model_name"] = "LSTM-softmax_2-10-2_d-19-10-2016_v-1.prm"

parameter["batch_size" ] = 5


parameter["data_location"] = "data_set/"
parameter["test_data_name"] = "little-timer_test.klepto"





model = rnnModel(parameter)

forward_fn = model.get_forward_function()

model.mbh.create_mini_batches("test")
test_mb_set_x, test_mb_set_y, test_mb_set_m = model.mbh.load_mini_batches("test")


###### TEST MODEL
ce_error = np.zeros([model.prm.data["test_batch_quantity"]*model.prm.data["batch_size"]])
auc_error = np.zeros([model.prm.data["test_batch_quantity"]*model.prm.data["batch_size"]])

for v in np.arange(0, model.prm.data["test_batch_quantity"]):
    v_net_out_ = forward_fn(test_mb_set_x[v], test_mb_set_m[v])[0]

    for b in np.arange(0,model.prm.data["batch_size"]):
        true_out = test_mb_set_y[v][:, b, :]
        code_out = v_net_out_[:, b, :]

        count = v * model.prm.data["batch_size"] + b

        ce_error[count] = sklearn.metrics.log_loss(true_out, code_out)
        auc_error[count] = sklearn.metrics.roc_auc_score(true_out, code_out)


print("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(ce_error)))
print("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error)))


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
plt.plot(test_mb_set_y[sample_no][:,batch,0], label="Target")
plt.plot(net_out[:,batch,0], label="LSTM Output")
plt.legend(loc='upper right',frameon=True)
plt.ylim([0,1.1])
plt.xlim([0,80])
plt.show()


### Delete mini batches
model.mbh.delete_mini_batches("test")
print("### TEST FINISH ###")