from __future__ import absolute_import, print_function, division
"""
This file contains a master class with support functions like load models, dump and print.
"""

######                           Imports
########################################
from abc import ABCMeta, abstractmethod
import klepto
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict
from future.utils import iteritems
import datetime
import time
import os.path
from .parameter_supervisor import ParameterSupervisor
from .mini_batch_handler import MiniBatchHandler


#### Master class with support and abstract methods
########################################
class ModelMaster(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.parameter = OrderedDict()


    ######       Create recnet model method
    ########################################
    def create(self, functions=None):

        self.prm = ParameterSupervisor()

        if "model_location" in self.parameter:
            if self.parameter["model_location"]:
                if not "model_location" in self.parameter:
                    raise Warning("Model loction is missing")
                if not "model_name" in self.parameter:
                    raise Warning("Model name is missing")

                old_weights, basic, struct,optimize = self.load(self.parameter["model_location"] + self.parameter["model_name"])

                self.prm.basic = basic
                self.prm.struct = struct
                self.prm.optimize = optimize

                self.prm.overwrite_parameter_dict(self.parameter)
                self.prm.pass_data_dict(self.parameter)

            else:
                self.prm.pass_all_parameter_dict(self.parameter)
                old_weights = np.repeat(None,self.prm.struct["net_size"].__len__())

        else:
            self.prm.pass_all_parameter_dict(self.parameter)
            old_weights = np.repeat(None,self.prm.struct["net_size"].__len__())

        self.generate_random_streams()
        self.make_modelname()

        self.mbh = MiniBatchHandler(self.rng, self.prm)
        self.mbh.check_out_data_set()
        self.pub("#mini batch handler: data check completed")


        self.build_model(old_weights)
        self.prm.struct["weight_numb"] = self.calc_numb_weights(self.all_weights)
        self.print_model_params()
        self.pub(" #-- Build model --#")

        if functions == None:
            self.pub(" #-- Compile all functions --#")
            self.compiling_training_function()
            self.compiling_validation_function()
            self.compiling_forward_function()
        elif all(x in ['train', 'valid', 'forward'] for x in functions):
            if 'train' in functions:
                self.pub(" #-- Compile train function --#")
                self.compiling_training_function()
            if 'valid' in functions:
                self.pub(" #-- Compile valid function --#")
                self.compiling_validation_function()
            if 'forward' in functions:
                self.pub(" #-- Compile forward function --#")
                self.compiling_forward_function()
        else:
            raise Warning("Please provide valid compile names ('train','valid','forward')")

        self.pub(" #-- Compilation complete --#")



    ######       Abstract build model method
    ########################################
    @abstractmethod
    def build_model(self, old_weights=None):
        pass

    ######       Abstract compiling methods
    ########################################
    @abstractmethod
    def compiling_training_function(self):
        pass

    @abstractmethod
    def compiling_validation_function(self):
        pass

    @abstractmethod
    def compiling_forward_function(self):
        pass

    ######           Generate random streams
    ########################################
    def generate_random_streams(self):
        self.rng = np.random.RandomState(self.prm.optimize["random_seed"])
        self.trng = RandomStreams(self.prm.optimize["random_seed"])


    ######                  Get mini batches
    ########################################
    def get_mini_batches(self, set_name):

        if set_name not in ['test', 'train', 'valid']:
            raise Warning("'get_mini_batches': Wrong set name")

        if self.prm.data["checked_data"][set_name] == False:
            raise Warning(set_name + " data set is not available")

        if "mb_of_" + self.prm.data[set_name + "_data_name"] in os.listdir(self.prm.data["mini_batch_location"][:-1]):
            self.mbh.delete_mini_batches(set_name)
            self.pub("#mini batch handler: delete old " + set_name + " mini batches")
            self.mbh.create_mini_batches(set_name)
            self.pub("#mini batch handler: create new " + set_name + " mini batches")
        else:
            self.mbh.create_mini_batches(set_name)
            self.pub("#mini batch handler: create new " + set_name + " mini batches")

        return self.mbh.load_mini_batches(set_name)


    ######                Clear mini batches
    ########################################
    def clear_mini_batches(self, set_name):

        if set_name not in ['test', 'train', 'valid']:
            raise Warning("'clear_mini_batches': Wrong set name")

        if "mb_of_" + self.prm.data[set_name + "_data_name"] in os.listdir(self.prm.data["mini_batch_location"][:-1]): #os.getcwd()):
            self.mbh.delete_mini_batches(set_name)
            self.pub("#mini batch handler# delete old " + set_name + " mini batches")

    def sample_quantity(self, set_name):
        return self.prm.data[set_name + "_batch_quantity"] *self.prm.data["batch_size"]


    def batch_quantity(self, set_name):
        return self.prm.data[set_name + "_batch_quantity"]

    def batch_size(self):
        return self.prm.data["batch_size"]

    def epochs(self):
        return self.prm.optimize["epochs"]


    ######       Dump weights in kelpto file
    ########################################
    def dump(self):

        data_location = self.prm.basic["model_location"] + self.prm.basic["model_name"] + ".prm"
        self.pub("Save model: " + data_location)
        d = klepto.archives.file_archive(data_location, cached=True,serialized=True)
        d['layer_weights'] = [[np.asarray(w.eval()) for w in layer] for layer in self.layer_weights]
        d['p_basic'] = self.prm.basic
        d['p_struct'] = self.prm.struct
        d['p_optimize'] = self.prm.optimize
        d.dump()
        d.clear()


    ######     Load weights form kelpto file
    ########################################
    def load(self, data_location):

        if not os.path.isfile(data_location):
            raise Warning("No saved parameters found")

        d = klepto.archives.file_archive(data_location, cached=True,serialized=True)
        d.load()
        weights = d['layer_weights']
        basic = d['p_basic']
        struct = d['p_struct']
        optimize= d['p_optimize']
        d.clear()

        return weights, basic, struct,optimize


    ######       Calculate number of weights
    ########################################
    def calc_numb_weights(self, weights):
        numb_weights = 0
        for w in weights:
            wn =  w.get_value().shape
            if wn.__len__() == 1:
                numb_weights = numb_weights + wn[0]
            if wn.__len__() == 2:
                numb_weights = numb_weights + wn[0] * wn[1]
        return numb_weights


    ######               Creates a file name
    ########################################
    def make_modelname(self):
        day_str = time.strftime("%d-%m-%Y")
        net_str = "-".join(str(e) for e in self.prm.struct["net_size"])
        type_str = "-".join([str(e) for e in self.prm.struct["net_unit_type"]][1:])
        if 'bi' in self.prm.struct["net_arch"]:
            bi_str = '_bi'
        else:
            bi_str = ''
        numb = 1
        fname =  type_str + "_" + net_str + bi_str + "_d-" + day_str + "_v-" + str(numb)

        while(os.path.isfile(self.prm.basic["output_location"] + fname + ".log")):
            numb +=1
            fname =  type_str + "_" + net_str + bi_str + "_d-" + day_str + "_v-" + str(numb)

        self.prm.basic["model_name"] =  fname


    ######      Manages write/print commands
    ########################################
    def pub(self, text):
        if(self.prm.basic["output_type"]=="console" or self.prm.basic["output_type"]=="both"):
            print(text)
        if(self.prm.basic["output_type"]=="file" or self.prm.basic["output_type"]=="both"):
            self.fobj = open(self.prm.basic["output_location"] + self.prm.basic["model_name"]  + ".log", "a")
            self.fobj.write(str(text) + "\n")
            self.fobj.close()


    ######           Print model parameters
    ########################################
    def print_model_params(self):
            self.pub("###")
            self.pub("####### RecNet - Recurrent Neural Network Framework ########")
            self.pub("###")
            date_time = str(datetime.datetime.today())
            self.pub("# Start Datetime: "+ date_time )
            self.pub("###")
            self.pub("# Basic Informations")
            for kk, pp in iteritems(self.prm.basic):
                str_obj = str(kk) + ": "+ str(pp)
                self.pub(str_obj)
            self.pub("###")
            self.pub("# Data Information")
            for kk, pp in iteritems(self.prm.data):
                str_obj = str(kk) + ": " + str(pp)
                self.pub(str_obj)
            self.pub("###")
            self.pub("# Network Structure")
            for kk, pp in iteritems(self.prm.struct):
                str_obj = str(kk) + ": "+ str(pp)
                self.pub(str_obj)
            self.pub("###")
            self.pub("# Optimization Parameters")
            for kk, pp in iteritems(self.prm.optimize):
                str_obj = str(kk) + ": "+ str(pp)
                self.pub(str_obj)
            self.pub("###")
