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

    def __init__(self, saccs, temperature=None):
        """
        Constructor of the HSCLikeModule
        """

        self.saccs = saccs

        if temperature is not None:
            logger.info('temperature = {}. Scaling all log-likelihoods by temperature.'.format(temperature))
            self.apply_temperature = True
            self.temperature = temperature

        else:
            logger.info('No temperature provided. Running unscaled log-likelihoods.')
            self.apply_temperature = False
            self.temperature = None


    def computeLikelihoodFromCl(self, cl_theory):
        """
        Computes the likelihood using information from the context
        """
        # Calculate a likelihood up to normalization
        lnprob = 0.
        for i, s in enumerate(self.saccs):
            delta = s.mean.vector - cl_theory[i]
            pmatrix = s.precision.getPrecisionMatrix()
            lnprob += np.einsum('i,ij,j',delta, pmatrix, delta)
        lnprob *= -0.5

        if self.apply_temperature:
            lnprob /= self.temperature

        # Return the likelihood
        print(" <><> Likelihood evaluated, lnprob = ",lnprob)
        return lnprob

    def computeLikelihood(self, ctx):
        """
        Computes the likelihood using information from the context
        """
        # Get information from the context. This can be results from a core
        # module or the parameters coming from the sampler
        cl_theory = ctx.get('cl_theory')
        return self.computeLikelihoodFromCl(cl_theory)

    def setup(self):
        """
        Sets up the likelihood module.
        Tasks that need to be executed once per run
        """
        #e.g. load data from files

