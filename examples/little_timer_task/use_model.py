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

### 1. Step: Create new model
rn = rnnModel()

### 2. Step: Define parameters
rn.parameter["load_model"] = True
rn.parameter["model_location"] = "model_save/"
################################################################################
########################### ADD NAME FROM TRAINED MODEL HERE ! #################
rn.parameter["model_name"] = "**********************************.prm"
rn.parameter["batch_size" ] = 5
rn.parameter["data_location"] = "data_set/"
rn.parameter["test_data_name"] = "little-timer_test.klepto"

### 3. Step: Create model and compile functions
rn.create(['forward'])


### 3. Step: Get mini batches from your test data set
rn.mbh.create_mini_batches("test")
test_mb_set_x, test_mb_set_y, test_mb_set_m = rn.mbh.load_mini_batches("test")


### 4. Step: Define your model test
cross_entropy_error = np.zeros([rn.prm.data["test_batch_quantity"]*rn.prm.data["batch_size"]])
auc_error = np.zeros([rn.prm.data["test_batch_quantity"]*rn.prm.data["batch_size"]])

#iterate over batches
for v in np.arange(0, rn.prm.data["test_batch_quantity"]):
    #get network output
    v_net_out_ = rn.forward_fn(test_mb_set_x[v], test_mb_set_m[v])[0]
    #iterate over batch size
    for b in np.arange(0,rn.prm.data["batch_size"]):
        #calculate error
        true_out = test_mb_set_y[v][:, b, :]
        code_out = v_net_out_[:, b, :]
        count = v * rn.prm.data["batch_size"] + b
        cross_entropy_error[count] = sklearn.metrics.log_loss(true_out, code_out)
        auc_error[count] = sklearn.metrics.roc_auc_score(true_out, code_out)

print("## cross entropy sklearn : " + "{0:.4f}".format(np.mean(cross_entropy_error)))
print("## area under the curve  : " + "{0:.4f}".format(np.mean(auc_error)))

# Plot results
sample_no = 0
batch = 0
net_out = rn.forward_fn(test_mb_set_x[sample_no], test_mb_set_m[sample_no])[0]

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
rn.mbh.delete_mini_batches("test")
print("### TEST FINISH ###")