#!/usr/bin/env python

""" Numbers recognition """
"""____________________"""
"""      USE MODEL
"""


######  Set global Theano config  #######
import os
t_flags = "mode=FAST_RUN,device=cpu,floatX=float32, optimizer='fast_run', allow_gc=False"
print "Theano Flags: " + t_flags
os.environ["THEANO_FLAGS"] = t_flags

######         Imports          ######
import numpy as np
import matplotlib.pyplot as plt
import recnet
from util import edit_distance

### 1. Step: Create new model
rn = recnet.rnnModel()

### 2. Step: Define parameters
rn.parameter["load_model"] = True
rn.parameter["model_location"] = "model_save/"
################################################################################
########################### ADD NAME FROM TRAINED MODEL HERE ! #################
rn.parameter["model_name"] = "***************************.prm"
rn.parameter["batch_size" ] = 1
rn.parameter["data_location"] = "data_set/"
rn.parameter["test_data_name"] = "numbers_image_test.klepto"

### 3. Step: Create model and compile functions
rn.create(['forward'])

### 4. Step: Get mini batches from test data set
mb_test_x, mb_test_y, mb_test_m = rn.get_mini_batches("test")

### 5. Step: Determine mean edit distance
test_error = np.zeros([rn.batch_quantity('test'),rn.batch_size()])

for j in xrange(rn.batch_quantity('test')):
    net_out = rn.forward_fn( mb_test_x[j], mb_test_m[j])

    for b in xrange(rn.batch_size()):
        true_out = mb_test_y[j][b, :]
        cln_true_out = np.delete(true_out, np.where(true_out == 10))

        net_out = net_out[:, b, :]
        arg_net_out = np.argmax(net_out, axis=1)
        cln_net_out = np.delete(arg_net_out, np.where(arg_net_out == 10))

        test_error[j, b] = edit_distance(cln_true_out, cln_net_out)

print("Test set mean edit distance: " + "{0:.4f}".format(np.mean(test_error)))

# Plot results
sample_no = 1
batch = 0
net_out = rn.forward_fn(mb_test_x[sample_no], mb_test_m[sample_no])
sample = mb_test_x[sample_no][:,batch,:]
mask = mb_test_m[sample_no][:,batch,:]
signal = net_out[:,batch,:]

fig = plt.figure()
fig.suptitle('Numbers recognition - Sample')
plt.subplot(2, 1, 1)
plt.xlabel('Image of numbers')
plt.imshow(sample.transpose())
plt.subplot(2, 1, 2)
plt.xlabel('CTC output signal')
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'blank']
for i in xrange(11):
    plt.plot(signal[:, i], label=labels[i])
plt.legend(loc='upper right', frameon=True)
plt.ylim([0, 1.1])
plt.xlim([0, int(mask[:, 0].sum())])
plt.show()

### Delete mini batches
rn.clear_mini_batches("test")
print("### TEST FINISH ###")