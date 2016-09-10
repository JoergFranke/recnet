from __future__ import print_function
__author__ = 'Joerg Franke'
"""
This file contains the organization of the mini-batches. It loads existing mini-batch-data and creates mini-batches from a
list of sequences. This includes bucketing, padding and mask creation.
"""

######                           Imports
########################################
import os
import klepto
import numpy as np
import theano


######                  Load minibatches
########################################
def load_minibatches(data_location, data_name, batch_size, recreate=False):

    # find existing minibatch sets
    set_exists = False
    mb_set_name = data_name+ "_mb" + str(int(batch_size)) + ".klepto"
    for root, dirs, files in os.walk(data_location):
        if mb_set_name in files:
            set_exists = True

    # if not, create minibatch set
    if set_exists == False or recreate==True:
        print("create minibatches, filename: " + mb_set_name)
        creat_minibatches(data_location, data_name, batch_size, mb_set_name)

    # load mini-batch-data-set
    print("load " + mb_set_name)
    file_name = data_location + mb_set_name
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d.load()
    data_mb_set_x = d['x']
    data_mb_set_y = d['y']
    data_mb_set_m = d['m']
    d.clear()
    print("set size: "+ str(data_mb_set_x.__len__()))

    return  data_mb_set_x, data_mb_set_y, data_mb_set_m


######                Create minibatches
########################################
def creat_minibatches(data_location, data_set_name, batch_size, mb_set_name):

    rng = np.random.RandomState(seed=1)

    # Load data set
    file_name = data_location + data_set_name + ".klepto"
    print(file_name)
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d.load()
    data_set_x = d['x']
    data_set_y = d['y']
    d.clear()

    # data set info
    len_set = data_set_x.__len__()
    print("len_set " + str(len_set))
    len_x = data_set_x[0].shape[1]

    nbatches = int(len_set / batch_size)


    len_y = data_set_y[0].shape[1]

    print(len_x)
    print(len_y)

    # permute data set
    permute_range = rng.permutation(np.arange(len_set))
    data_set_x_p = []
    data_set_y_p = []
    for i in permute_range:
        data_set_x_p.append(data_set_x[i])
        data_set_y_p.append(data_set_y[i])

    # get sequence lengths
    len_seq = np.empty(len_set)
    for s in xrange(len_set):
        len_seq[s] = data_set_x_p[s].shape[0]
    print("Sequence length (min, q25, mean, q75, max)")
    print(len_seq.min(), np.percentile(len_seq, 25), len_seq.mean() , np.percentile(len_seq, 75), len_seq.max())

    # sort index by length
    sort_index = np.argsort(len_seq)
    data_set_x_s = []
    data_set_y_s = []
    for i in sort_index:
        data_set_x_s.append(data_set_x_p[i])
        data_set_y_s.append(data_set_y_p[i])

    # create minibatches
    print(" create minibatches")
    data_mb_x = []
    data_mb_y = []
    data_mask = []


    for i in np.arange(nbatches):
        # max batch size = 1000
        mb_size = np.min([data_set_x_s[min(i*batch_size+batch_size-1,len_set)].shape[0], 1000])

        mb_train_x = np.zeros([mb_size, batch_size, len_x])
        mb_train_y = np.zeros([mb_size, batch_size, len_y])
        mb_mask = np.zeros([mb_size, batch_size, 1])

        for s in np.arange(batch_size):
            len_sample =  np.min([data_set_x_s[i*batch_size+s].shape[0], 1000])
            mb_train_x[:len_sample,s,:] = data_set_x_s[i*batch_size+s][:len_sample]
            mb_train_y[:len_sample,s,:] = data_set_y_s[i*batch_size+s][:len_sample]
            mb_mask[:len_sample,s,:] = np.ones([len_sample,1])


        data_mb_x.append(mb_train_x.astype(theano.config.floatX))
        data_mb_y.append(mb_train_y.astype(theano.config.floatX))
        data_mask.append(mb_mask.astype(theano.config.floatX))

    data_mb_x = np.asarray(data_mb_x)
    data_mb_y = np.asarray(data_mb_y)
    data_mask = np.asarray(data_mask)

    # Permutate train set
    data_set_length = data_mb_x.__len__()
    order = np.arange(0,data_set_length)
    order = rng.permutation(order)
    data_mb_x = data_mb_x[order]
    data_mb_y = data_mb_y[order]
    data_mask = data_mask[order]

        
    print("write minibatch set")
    print("minibatch set length: " + str(data_mb_x.__len__()))
    file_name = data_location + mb_set_name
    print("minibatch set name: " + file_name)
    d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
    d['x'] = data_mb_x
    d['y'] = data_mb_y
    d['m'] = data_mask
    d.dump()
    d.clear()

    return

