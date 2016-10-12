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


class prmSupervisor:

    def __init__(self):
        self.basic = OrderedDict()
        self.struct = OrderedDict()
        self.optimize = OrderedDict()


    def organize_prm(self):



        self.p_struct["file_name"] = self.make_filename()
        self.p_struct["weight_numb"] = self.__calc_numb_weights(self.all_weights)















