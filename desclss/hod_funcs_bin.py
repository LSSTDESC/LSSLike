import numpy as np
import pyccl as ccl
# import logging
# logging.basicConfig(level=logging.INFO)

class HODParams(object):

    def __init__(self, hodpars, islogm0=False, islogm1=False):

        # self.log = logging.getLogger('HODParams')
        # self.log.setLevel(logging.INFO)
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(levelname)s: %(message)s')
        # ch.setFormatter(formatter)
        # self.log.addHandler(ch)
        # self.log.propagate = False

        self.params = hodpars
        if islogm0:
            self.params['m0_0'] = 10**self.params['m0_0']
        if islogm1:
            self.params['m1_0'] = 10**self.params['m1_0']
        #print('Parameters updated: hodpars = {}.'.format(hodpars))

        return

    def lmminf(self, z) :
        #Returns log10(M_min)
        lmmin = self.params['lmmin_0']
        return lmmin

    def sigmf(self, z):
        sigm = self.params['sigm_0']
        return sigm

    def m0f(self, z) :
        # Returns M_0
        m0 = self.params['m0_0']
        return m0

    def m1f(self, z) :
        #Returns M_1
        m1 = self.params['m1_0']
        return m1

    def alphaf(self, z) :
        #Returns alpha
        alpha = self.params['alpha_0']
        return alpha

    def fcf(self, z) :
        #Returns f_central
        fc = self.params['fc_0']
        return fc
