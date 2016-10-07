__author__ = 'Joerg Franke'
"""
This file contains a master class with support functions like load models, dump and print.
"""

######                           Imports
########################################
import klepto
import numpy as np
from collections import OrderedDict
import datetime
import time
import os.path
import sys



#### Master class with support functions
########################################
class modelMaster:

    def __init__(self, p_struct, p_optima, rng, trng, load=False, data_location=None, new_batch_size=None):

        if not load:
            self.build_model(p_struct, p_optima, rng, trng, np.repeat(None, p_struct['hidden_layer']+1))
            # self.p_struct = p_struct
            # self.p_optima = p_optima
            self.p_struct["file_name"] = self.make_filename()
            self.p_struct["weight_numb"] = self.__calc_numb_weights(self.all_weights)

        else:
            print("load params")
            old_weights, struct, optima = self.load(data_location)

            if new_batch_size != None:
                struct["batch_size"] = new_batch_size


            self.build_model(struct, optima, rng, trng, old_weights)

            self.p_struct["file_name"] = self.make_filename()
            self.pub("load " + data_location)


    ######       Dump weights in kelpto file
    ########################################
    def dump(self):

        data_location = self.p_struct["file_name"] + ".prm"
        self.pub("save " + data_location)
        d = klepto.archives.file_archive(data_location, cached=True,serialized=True)
        d['layer_weights'] = [[np.asarray(w.eval()) for w in layer] for layer in self.layer_weights]
        d['p_struct'] = self.p_struct
        d['p_optima'] = self.p_optima
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
        struct = d['p_struct']
        optima = d['p_optima']
        d.clear()

        return weights, struct, optima


    ######       Calculate number of weights
    ########################################
    def __calc_numb_weights(self, weights):
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
        net_str = "-".join(str(e) for e in self.p_struct["net_size"])
        if(self.p_struct["bi_directional"]):
            bi_str = '-bi'
        else:
            bi_str = ''
        numb = 1
        fname =  self.p_struct['output_location'] + "n-" + net_str + bi_str + "_d-" + day_str + "_v-" + str(numb)

        while(os.path.isfile(fname + ".log")):
            numb +=1
            fname =  self.p_struct['output_location'] + "n-" + net_str + bi_str + "_d-" + day_str + "_v-" + str(numb)

        return fname


    ######      Manages write/print commands
    ########################################
    def pub(self, text):
        if(self.p_struct["output_type"]=="console" or self.p_struct["output_type"]=="both"):
            print text
        if(self.p_struct["output_type"]=="file" or self.p_struct["output_type"]=="both"):
            self.fobj = open(self.p_struct["file_name"]  + ".log", "a")
            self.fobj.write(str(text) + "\n")
            self.fobj.close()


    ######           Writes model parameters
    ########################################
    def print_model_params(self):
        if(self.p_struct["output_type"]=="console" or self.p_struct["output_type"]=="both"):
            print "###"
            print "####### Boundary Detection with use of Long Short term Memories ########"
            print "###"
            print "# Start Datetime: ", datetime.datetime.today()
            print "###"
            print "# Network Structure"
            for kk, pp in self.p_struct.iteritems():
                print kk, ": ", pp
            print "###"
            print "# Optimization Parameters"
            for kk, pp in self.p_optima.iteritems():
                print kk, ": ", pp
            print "###"
        if(self.p_struct["output_type"]=="file" or self.p_struct["output_type"]=="both"):
            self.fobj = open(self.p_struct["file_name"] + ".log", "a")

            self.fobj.write( "###" + "\n")
            self.fobj.write( "####### Boundary Detection with use of Long Short term Memories ########" + "\n")
            self.fobj.write( "###" + "\n")
            date_time = str(datetime.datetime.today())
            self.fobj.write("# Start Datetime: "+ date_time + "\n" )
            self.fobj.write( "###" + "\n")
            self.fobj.write( "# Network Structure" + "\n")
            for kk, pp in self.p_struct.iteritems():
                str_obj = str(kk) + ": "+ str(pp)
                self.fobj.write(str_obj+ "\n")
            self.fobj.write( "###" + "\n")
            self.fobj.write( "# Optimization Parameters" + "\n")
            for kk, pp in self.p_optima.iteritems():
                str_obj = str(kk) + ": "+ str(pp)
                self.fobj.write(str_obj+ "\n")
            self.fobj.write("###" + "\n")
            self.fobj.close()


######    Supervise structure parameters
########################################
    def prm_structure_supervise(self, prm_given):

        prm_structure = OrderedDict()

        if prm_given["batch_size"] > 0:
            prm_structure["batch_size"] = prm_given
        else:
            #self.pub("Batch size below 1")
            sys.exit("Parameter error")

        if (sys.version_info[0] == 2 and  isinstance(prm_given["corpus_name"   ], basestring)) or \
            (sys.version_info[0] == 3 and isinstance(prm_given["corpus_name"   ], str)):
            prm_structure["corpus_name"   ] = prm_given["corpus_name"   ]
        else:
            #self.pub("Batch size below 1")
            sys.exit("Parameter error")

        # todo add tests

        prm_structure["data_location" ] = prm_given["data_location" ]
        prm_structure["net_size"      ] = prm_given["net_size"      ]
        prm_structure["net_unit_type" ] = prm_given["net_unit_type" ]
        prm_structure["hidden_layer"  ] = prm_given["hidden_layer"  ]
        prm_structure["bi_directional"] = prm_given["bi_directional"]
        prm_structure["identity_func" ] = prm_given["identity_func" ]
        prm_structure["train_set_len" ] = prm_given["train_set_len" ]
        prm_structure["valid_set_len" ] = prm_given["valid_set_len" ]
        if "log" not in os.listdir(os.getcwd()):
            os.mkdir("log")
        prm_structure["output_location"] = prm_given["output_location"]
        prm_structure["output_type"    ] = prm_given["output_type"    ]

        return prm_structure