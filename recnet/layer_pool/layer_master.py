__author__ = 'joerg'

######                           Imports
########################################
from abc import ABCMeta, abstractmethod
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

class LayerMaster(object):

    __metaclass__ = ABCMeta


    ###### Abstract sequenceiteration method
    ########################################
    @abstractmethod
    def sequence_iteration(self):
        pass



#
# class unidirectional_layer
#
#
# class bidirectional_layer