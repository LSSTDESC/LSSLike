#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HSCLikeModule(object):
    """
    Dummy object for calculating a likelihood
    """

    def __init__(self, saccs):
        """
        Constructor of the HSCLikeModule
        """

        self.saccs = saccs

    def computeLikelihood(self, ctx):
        """
        Computes the likelihood using information from the context
        """
        # Get information from the context. This can be results from a core
        # module or the parameters coming from the sampler
        cl_theory = ctx.get('cl_theory')

        # Calculate a likelihood up to normalization
        lnprob = 0.
        for i, s in enumerate(self.saccs):
            delta = s.mean.vector - cl_theory[i]
            pmatrix = s.precision.getPrecisionMatrix()
            lnprob += np.einsum('i,ij,j',delta, pmatrix, delta)
        lnprob *= -0.5

        # Return the likelihood
        return lnprob

    def setup(self):
        """
        Sets up the likelihood module.
        Tasks that need to be executed once per run
        """
        #e.g. load data from files

