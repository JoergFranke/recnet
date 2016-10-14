from __future__ import print_function
__author__ = 'Joerg Franke'
"""
This file contains the organization of the mini-batches. It loads existing mini-batch-data and creates mini-batches from a
list of sequences/file. This includes bucketing, padding and mask creation.
"""

######                           Imports
########################################
import os
import klepto
import numpy as np
import theano


class MiniBatchHandler:

    def __init__(self,rng, prm_data, prm_struct):
        self.rng
        self.prm_data = prm_data
        self.prm_struct = prm_struct

    def check_out_data_set(self):
        file_name = self.prm_data["data_location"] + self.prm_data["train_data_name"]
        try:
            d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
            d.load()
            data_set_x = d['x']
            data_set_y = d['y']
            d.clear()
            self.prm_data["train_set_len"] = data_set_x.__len__()
            if data_set_x.__len__() != data_set_y.__len__():
                raise Warning("x and y train_data_name have not the same length")
            self.prm_data["x_size"] = data_set_x[0].shape[1]
            if self.prm_data["x_size"] != int(self.prm_struct["net_size"][0]):
                raise Warning("train data x size and net input size are unequal")
            self.prm_data["y_size"] = data_set_y[0].shape[1]
            if self.prm_data["y_size"] != int(self.prm_struct["net_size"][-1]):
                raise Warning("train data y size and net input size are unequal")
            del data_set_x
            del data_set_x
        except KeyError:
            raise Warning("data_location or train_data_name wrong")

        file_name = self.prm_data["data_location"] + self.prm_data["valid_data_name"]
        try:
            d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
            d.load()
            data_set_x = d['x']
            data_set_y = d['y']
            d.clear()
            self.prm_data["valid_set_len"] = data_set_x.__len__()
            if data_set_x.__len__() != data_set_y.__len__():
                raise Warning("x and y valid_data_name have not the same length")
            self.prm_data["x_size"] = data_set_x[0].shape[1]
            if self.prm_data["x_size"] != int(self.prm_struct["net_size"][0]):
                raise Warning("valid data x size and net input size are unequal")
            self.prm_data["y_size"] = data_set_y[0].shape[1]
            if self.prm_data["y_size"] != int(self.prm_struct["net_size"][-1]):
                raise Warning("valid data y size and net input size are unequal")
        except KeyError:
            raise Warning("data_location or valid_data_name wrong")

        self.prm_data["checked_data"] = True


    def create_mini_batches(self, set):

        if self.prm_data["checked_data"] == False:
            self.check_out_data_set()

        if set != "train" and set != "valid":
            raise Warning("set must be train or valid")

        file_name = self.prm_data["data_location"] + self.prm_data[set + "_data_name"]
        d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
        d.load()
        data_set_x = d['x']
        data_set_y = d['y']
        d.clear()

        self.prm_data[set + "_data_x_len"] = [i.__len__() for i in data_set_x]
        self.prm_data[set + "_data_y_len"]= [i.__len__() for i in data_set_y]
        if not np.array_equal(self.prm_data[set + "_data_x_len"], self.prm_data[set + "_data_y_len"]):
            raise Warning(set + " x and y sequences have not the same length")

        self.prm.data[set + "_batch_quantity"] = int(np.trunc(self.prm_data[set + "_set_len" ]/self.prm_data["batch_size"]))


        sample_order = np.arange(0,self.prm_data[set + "_set_len" ])
        sample_order = self.rng.permutation(sample_order)

        data_mb_x = []
        data_mb_y = []
        data_mask = []

        for j in xrange(model.prm.data["batch_quantity"]):

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

            data_mb_x.append(mb_train_x.astype(theano.config.floatX))
            data_mb_y.append(mb_train_y.astype(theano.config.floatX))
            data_mask.append(mb_mask.astype(theano.config.floatX))

        data_mb_x = np.asarray(data_mb_x)
        data_mb_y = np.asarray(data_mb_y)
        data_mask = np.asarray(data_mask)

        #todo generate and save mb name

        file_name = data_location + mb_set_name
        d = klepto.archives.file_archive(file_name, cached=True,serialized=True)
        d['x'] = data_mb_x
        d['y'] = data_mb_y
        d['m'] = data_mask
        d.dump()
        d.clear()

        #todo wirte batch quantity,
        self.prm_data["data_location"]




    def delete_mini_batches(self, set):
        pass

    def load_mini_batches(self, set):



        return mb_train_x, mb_train_y, mb_train_m