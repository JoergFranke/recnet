__author__ = 'Joerg Franke'
"""
This file contains a supervision for parameters and contains all dictionaries (basic, data, structure, optimization)
which are necessary to build and train a model.
"""

######                           Imports
########################################
import numpy as np
from collections import OrderedDict
import datetime
import time
import os.path
import sys


######        Parameter Supervisor Class
########################################
class ParameterSupervisor:

    def __init__(self):
        self.basic = OrderedDict()
        self.data = OrderedDict()
        self.struct = OrderedDict()
        self.optimize = OrderedDict()


    ###### Method for testing strings an locations (under construction)
    ########################################
    def is_string(self, string):
        if (sys.version_info[0] == 2 and  isinstance(string, basestring)) or \
            (sys.version_info[0] == 3 and isinstance(string, str)):
            return True
        else:
            return False


    ###### Passes parameters in all dictionaries
    ########################################
    def pass_all_parameter_dict(self, parameter):
        self.pass_basic_dict(parameter)
        self.pass_data_dict(parameter)
        self.pass_structure_dict(parameter)
        self.pass_optimize_dict(parameter)


    ###### Overwrite old parameters with new ones
    ########################################
    def overwrite_parameter_dict(self, parameter):
        for kk, pp in parameter.iteritems():
            if kk in self.basic:
                self.basic[kk] = pp
            if kk in self.data:
                self.data[kk] = pp
            if kk in self.struct:
                self.struct[kk] = pp
            if kk in self.optimize:
                self.optimize[kk] = pp


    ###### Passes parameters in basic dictionary
    ########################################
    def pass_basic_dict(self, prm_basic):

        if "output_type" in prm_basic:
            self.basic["output_type"] = prm_basic["output_type"]
        else:
            self.basic["output_type"] = "both"

        if "output_location" in prm_basic:
            self.basic["output_location"] = prm_basic["output_location"]
        else:
            self.basic["output_location"] = "log/"

        if self.basic["output_location"][:-1] not in os.listdir(os.getcwd()):
            os.mkdir(self.basic["output_location"])

        if "model_location" in prm_basic:
            self.basic["model_location"] = prm_basic["model_location"]
        else:
            self.basic["model_location"] = "model_save/"

        if self.basic["model_location"][:-1] not in os.listdir(os.getcwd()):
            os.mkdir(self.basic["model_location"])

        if "load_model" in prm_basic:
            self.basic["load_model"] = prm_basic["load_model"]
            if self.basic["load_model"] == True:
                if "model_location" in prm_basic:
                    self.basic["model_location"] = prm_basic["model_location"]
                else:
                    raise Warning("Model loction is missing")

                if "model_name" in prm_basic:
                    self.basic["model_name"] = prm_basic["model_name"]
                else:
                    raise Warning("Model name is missing")
        else:
            self.basic["load_model"] = False


    ###### Passes parameters in data dictionary
    ########################################
    def pass_data_dict(self, prm_data):

        if "batch_size" in prm_data:
            if prm_data["batch_size"] > 0:
                self.data["batch_size"] = prm_data["batch_size"]
            else:
                raise Warning("Wrong batch size")
        else:
            raise Warning("No batch size")

        if "data_location" in prm_data:
            self.data["data_location"] = prm_data["data_location"]
        else:
            raise Warning("data_location is missing")

        if "train_data_name" in prm_data:
            self.data["train_data_name"] = prm_data["train_data_name"]
        else:
            self.data["train_data_name"] = None

        if "valid_data_name" in prm_data:
            self.data["valid_data_name"] = prm_data["valid_data_name"]
        else:
            self.data["valid_data_name"] = None

        if "test_data_name" in prm_data:
            self.data["test_data_name"] = prm_data["test_data_name"]
        else:
            self.data["test_data_name"] = None

        self.data["train_set_len" ] = 0
        self.data["valid_set_len" ] = 0
        self.data["test_set_len" ] = 0
        self.data["x_size"] = 0
        self.data["y_size"] = 0
        self.data["checked_data"] = {'train': False, 'valid': False, 'test': False}

        if "mini_batch_location" in prm_data:
            self.data["mini_batch_location"] = prm_data["mini_batch_location"]
        else:
            self.data["mini_batch_location"] = "mini_batch/"

        if self.data["mini_batch_location"][:-1] not in os.listdir(os.getcwd()):
            os.mkdir(self.data["mini_batch_location"])


    ##### Passes parameters in structure dictionary
    ########################################
    def pass_structure_dict(self, prm_structure):

        if "net_size" in prm_structure:
            self.struct["net_size"      ] = prm_structure["net_size"]
            self.struct["hidden_layer"  ] = prm_structure["net_size"].__len__() - 2
        else:
            raise Warning("No net size")

        if "net_unit_type" in prm_structure:
            self.struct["net_unit_type"      ] = prm_structure["net_unit_type"]
            if  prm_structure["net_unit_type"].__len__() != self.struct["net_size" ].__len__():
                raise Warning("Net size and unit type have no equal length")
        else:
            raise Warning("No net unit type")

        if "net_act_type" in prm_structure:
            self.struct["net_act_type"      ] = prm_structure["net_act_type"]
            if  prm_structure["net_act_type"].__len__() != self.struct["net_size" ].__len__():
                raise Warning("Net size and act type have no equal length")
        else:
            self.struct["net_act_type" ] = ['tanh' for i in xrange(prm_structure["net_size"].__len__())]

        if "net_arch" in prm_structure:
            self.struct["net_arch"      ] = prm_structure["net_arch"]
            if  prm_structure["net_arch"].__len__() != self.struct["net_size" ].__len__():
                raise Warning("Net size and net architecture have no equal length")
        else:
            raise Warning("No network architecture 'net_arch' ")

        self.struct["weight_numb"] = 0

        # if "weight_initialize" in prm_structure:
        #     self.struct["weight_initialize"] = prm_structure["weight_initialize"]
        # else:
        #     self.struct["weight_initialize"] = "uniform_sqrt"

        # if "bi_directional" in prm_structure:
        #     self.struct["bi_directional"] = prm_structure["bi_directional"]
        # else:
        #     self.struct["bi_directional"] = False todo tidy up

        if "identity_func" in prm_structure: #(currently corrupted)
            self.struct["identity_func"] = prm_structure["identity_func"]
        else:
            self.struct["identity_func"] = False


    ##### Passes parameters in optimize dictionary
    ########################################
    def pass_optimize_dict(self, prm_optimization):

        if "random_seed" in prm_optimization:
            self.optimize["random_seed"] = prm_optimization["random_seed"]
        else:
            self.optimize["random_seed"] = 1234

        if "epochs" in prm_optimization:
            self.optimize["epochs"] = prm_optimization["epochs"]
        else:
            raise Warning("Number of epochs is missing")

        if "optimization" in prm_optimization:
            if prm_optimization["optimization"] == "sgd":
                self.optimize["optimization"] = prm_optimization["optimization"]
                if not "learn_rate" in prm_optimization:
                    raise Warning("learn_rate is missing")

            if prm_optimization["optimization"] == "rmsprop":
                self.optimize["optimization"] = prm_optimization["optimization"]
                if not "learn_rate" in prm_optimization:
                    raise Warning("learn_rate is missing")
                if not "decay_rate" in prm_optimization:
                    raise Warning("decay_rate is missing")

            if prm_optimization["optimization"] == "momentum":
                self.optimize["optimization"] = prm_optimization["optimization"]
                if not "learn_rate" in prm_optimization:
                    raise Warning("learn_rate is missing")
                if not "momentum_rate" in prm_optimization:
                    raise Warning("momentum_rate is missing")

            if prm_optimization["optimization"] == "nesterov_momentum":
                self.optimize["optimization"] = prm_optimization["optimization"]
                if  not"learn_rate" in prm_optimization:
                    raise Warning("learn_rate is missing")
                if not "momentum_rate" in prm_optimization:
                    raise Warning("momentum_rate is missing")

            if prm_optimization["optimization"] == "nm_rmsprop":
                self.optimize["optimization"] = prm_optimization["optimization"]
                if "learn_rate" in prm_optimization:
                    raise Warning("learn_rate is missing")
                if not "momentum_rate" in prm_optimization:
                    raise Warning("momentum_rate is missing")
                if not "decay_rate" in prm_optimization:
                    raise Warning("decay_rate is missing")

            if prm_optimization["optimization"] == "adadelta":
                self.optimize["optimization"] = prm_optimization["optimization"]

            else:
                raise Warning("Name of optimization is wrong")
        else:
            raise Warning("Name of optimization is missing")

        if "learn_rate" in prm_optimization:
            self.optimize["learn_rate"] = prm_optimization["learn_rate"]
        else:
            self.optimize["learn_rate"] = 1

        if "momentum_rate" in prm_optimization:
            self.optimize["momentum_rate"] = prm_optimization["momentum_rate"]
        else:
            self.optimize["momentum_rate"] = 1

        if "decay_rate" in prm_optimization:
            self.optimize["decay_rate"] = prm_optimization["decay_rate"]
        else:
            self.optimize["decay_rate"] = 1

        if "use_dropout" in prm_optimization:
            self.optimize["use_dropout"] = prm_optimization["use_dropout"]

            if prm_optimization["use_dropout"] == True:
                if "dropout_level" in prm_optimization:
                    self.optimize["dropout_level"] = prm_optimization["dropout_level"]
                else:
                    raise Warning("Please quote dropout_level")
            else:
                self.optimize["dropout_level"] = 0
        else:
            self.optimize["use_dropout"] = False


        if "regularization" in prm_optimization:
            self.optimize["regularization"] = prm_optimization["regularization"]

            if prm_optimization["regularization"] == True:
                if "reg_factor" in prm_optimization:
                    self.optimize["reg_factor"] = prm_optimization["reg_factor"]
                else:
                    raise Warning("Please quote reg_factor")
        else:
            self.optimize["regularization"] = False


        if "noisy_input" in prm_optimization:
            self.optimize["noisy_input"] = prm_optimization["noisy_input"]

            if prm_optimization["noisy_input"] == True:
                if "noise_level" in prm_optimization:
                    self.optimize["noise_level"] = prm_optimization["noise_level"]
                else:
                    raise Warning("Please quote noise_level")
        else:
            self.optimize["noisy_input"] = False


        if "loss_function" in prm_optimization:
            self.optimize["loss_function"] = prm_optimization["loss_function"]

            if prm_optimization["loss_function"] == "w2_cross_entropy":
                if "bound_weight" in prm_optimization:
                    self.optimize["bound_weight"] = prm_optimization["bound_weight"]
                else:
                    raise Warning("Please quote bound_weight")
            else:
                self.optimize["bound_weight"] = 0
        else:
            raise Warning("Name of loss function is missing")






