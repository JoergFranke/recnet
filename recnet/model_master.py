__author__ = 'Joerg Franke'
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
import datetime
import time
import os.path
import sys
from parameter_supervisor import ParameterSupervisor
from mini_batch_handler import MiniBatchHandler


#### Master class with support functions
########################################
class ModelMaster(object):

    __metaclass__ = ABCMeta

    def __init__(self, parameter):


        self.prm = ParameterSupervisor()

        self.pass_parameter_dict(parameter)

        self.generate_random_streams()

        self.build_model()

        self.prm.struct["weight_numb"] = self.calc_numb_weights(self.all_weights)

        self.print_model_params()



        self.mbh = MiniBatchHandler(self.rng, self.prm.data, self.prm.struct, self.pub)


        if self.prm.basic["load_model"]:

            print("load params") #todo
            old_weights, struct, optima = self.load(data_location)

            #if new_batch_size != None:
            #    struct["batch_size"] = new_batch_size


            self.build_model(struct, optima, old_weights)

            self.prm.basic["file_name"] = self.make_filename()
            #self.pub("load " + data_location)

    @abstractmethod
    def build_model(self, old_weights=None):
        pass


    def pass_parameter_dict(self, parameter):
        self.prm.pass_basic_dict(parameter)
        self.prm.pass_data_dict(parameter)
        self.prm.pass_structure_dict(parameter)
        self.prm.pass_optimize_dict(parameter)

        self.prm.basic["file_name"] = self.make_filename()



    def generate_random_streams(self):
        self.rng = np.random.RandomState(self.prm.optimize["random_seed"])
        self.trng = RandomStreams(self.prm.optimize["random_seed"])




    ######       Dump weights in kelpto file
    ########################################
    def dump(self):

        data_location = self.prm.basic["file_name"] + ".prm"
        self.pub("save " + data_location)
        d = klepto.archives.file_archive(data_location, cached=True,serialized=True)
        d['layer_weights'] = [[np.asarray(w.eval()) for w in layer] for layer in self.layer_weights]
        d['p_basic'] = self.prm.basic
        d['p_data'] = self.prm.data
        d['p_struct'] = self.prm.struct
        d['p_optimize'] = self.prm.optimize
        #d['monitor'] = self.training_error
        d.dump()
        d.clear()


    ######     Load weights form kelpto file
    ########################################
    def load(self, data_location):

        assert os.path.isfile(data_location), "not saved parameters found"

        d = klepto.archives.file_archive(data_location, cached=True,serialized=True)
        d.load()
        weights = d['layer_weights']
        basic   = d['p_basic'   ]
        data   = d['p_data'    ]
        struct  = d['p_struct'  ]
        optimize= d['p_optimize']
        d.clear()
        # todo
        return weights, basic, data,struct,optimize


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
    def make_filename(self):
        day_str = time.strftime("%d-%m-%Y")
        net_str = "-".join(str(e) for e in self.prm.struct["net_size"])
        if(self.prm.struct["bi_directional"]):
            bi_str = '-bi'
        else:
            bi_str = ''
        numb = 1
        fname =  self.prm.basic['output_location'] + "n-" + net_str + bi_str + "_d-" + day_str + "_v-" + str(numb)

        while(os.path.isfile(fname + ".log")):
            numb +=1
            fname =  self.prm.basic['output_location'] + "n-" + net_str + bi_str + "_d-" + day_str + "_v-" + str(numb)

        return fname


    ######      Manages write/print commands
    ########################################
    def pub(self, text):
        if(self.prm.basic["output_type"]=="console" or self.prm.basic["output_type"]=="both"):
            print text
        if(self.prm.basic["output_type"]=="file" or self.prm.basic["output_type"]=="both"):
            self.fobj = open(self.prm.basic["file_name"]  + ".log", "a")
            self.fobj.write(str(text) + "\n")
            self.fobj.close()


    ######           Writes model parameters
    ########################################
    def print_model_params(self): #todo simplify
        if(self.prm.basic["output_type"]=="console" or self.prm.basic["output_type"]=="both"):
            print "###"
            print "####### RecNet - Recurrent Neural Network Framework ########"
            print "###"
            print "# Start Datetime: ", datetime.datetime.today()
            print "###"
            print "# Network Data"
            for kk, pp in self.prm.data.iteritems():
                print kk, ": ", pp
            print "###"
            print "# Network Structure"
            for kk, pp in self.prm.struct.iteritems():
                print kk, ": ", pp
            print "###"
            print "# Optimization Parameters"
            for kk, pp in self.prm.optimize.iteritems():
                print kk, ": ", pp
            print "###"
        if(self.prm.basic["output_type"]=="file" or self.prm.basic["output_type"]=="both"):
            self.fobj = open(self.prm.basic["file_name"] + ".log", "a")

            self.fobj.write( "###" + "\n")
            self.fobj.write( "####### RecNet - Recurrent Neural Network Framework ########" + "\n")
            self.fobj.write( "###" + "\n")
            date_time = str(datetime.datetime.today())
            self.fobj.write("# Start Datetime: "+ date_time + "\n" )
            self.fobj.write( "###" + "\n")
            self.fobj.write( "# Network Structure" + "\n")
            for kk, pp in self.prm.struct.iteritems():
                str_obj = str(kk) + ": "+ str(pp)
                self.fobj.write(str_obj+ "\n")
            self.fobj.write( "###" + "\n")
            self.fobj.write( "# Optimization Parameters" + "\n")
            for kk, pp in self.prm.optimize.iteritems():
                str_obj = str(kk) + ": "+ str(pp)
                self.fobj.write(str_obj+ "\n")
            self.fobj.write("###" + "\n")
            self.fobj.close()


