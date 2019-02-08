#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import pyccl as ccl
from scipy.interpolate import interp1d
import numpy as np
from desclss import hod
from cosmoHammer.exceptions import LikelihoodComputationException

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HSCCoreModule(object):
    """
    Dummy Core Module for calculating the squares of parameters.
    """

    def __init__(self, PARAM_MAPPING, DEFAULT_PARAMS, cl_params, saccs, noise, fid_cosmo_params=None):
        """
        Constructor of the DummyCoreModule
        """

        self.mapping = PARAM_MAPPING
        self.constants = DEFAULT_PARAMS
        self.cl_params = cl_params
        self.saccs = saccs
        self.noise = noise
        self.lmax = self.saccs[0].binning.windows[0].w.shape[0]
        self.ells = np.arange(self.lmax)
        if fid_cosmo_params is not None:
            self.cosmo = ccl.Cosmology(**fid_cosmo_params)

    def __call__(self, ctx):
        """
        Computes something and stores it in the context
        """
        # Get the parameters from the context
        p = ctx.getParams()

        params = self.constants.copy()
        for k,v in self.mapping.items():
            params[k] = p[v]

        # Calculate something
        cl_theory = [np.zeros((s.size(),)) for s in self.saccs]

        cosmo_params = self.get_params(params, 'cosmo')

        try:
            if (cosmo_params.viewkeys() & self.mapping.viewkeys()) != set([]):
                cosmo = ccl.Cosmology(**cosmo_params)
            else:
                cosmo = self.cosmo
            for i, s in enumerate(self.saccs):
                tracers = self.get_tracers(s, cosmo, params)

                if self.cl_params['fitHOD'] == 1 and self.cl_params['modHOD'] == 'zevol':
                    dic_hodpars = self.get_params(params, 'hod_'+self.cl_params['modHOD'])
                    self.hodpars = hod_funcs.HODParams(dic_hodpars, islogm0_0=True, islogm1_0=True)

                if self.cl_params['modHOD'] == 'zevol':
                    hodprof = hod.HODProfile(cosmo, self.hodpars.lmminf, self.hodpars.sigmf, self.hodpars.fcf, self.hodpars.m0f, \
                                                 self.hodpars.m1f, self.hodpars.alphaf)
                    # Provide a, k grids
                    pk_hod_arr = np.log(np.array([hodprof.pk(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr]))
                    pk_hod = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=pk_hod_arr, is_logp=True)

                for i1, i2, _, ells_binned, ndx in s.sortTracers() :
                    lmax_this=int(np.amax(ells_binned))
                    cls=np.zeros(len(self.ells))
                    if self.cl_params['modHOD'] is None:
                        logger.info('modHOD = {}. Not using HOD to compute theory predictions.'.format(self.cl_params['modHOD']))
                        cls[:lmax_this+1] = ccl.angular_cl(cosmo, tracers[i1], tracers[i2], np.arange(lmax_this+1))
                    elif self.cl_params['modHOD'] == 'zevol':
                        logger.info('modHOD = {}. Using HOD to compute theory predictions.'.format(self.cl_params['modHOD']))
                        cls[:lmax_this+1] = ccl.angular_cl(cosmo, tracers[i1], tracers[i2], np.arange(lmax_this+1), p_of_k_a=pk_hod)
                    elif self.cl_params['modHOD'] == 'bin':
                        dic_hodpars = self.get_params(params, 'hod_'+self.cl_params['modHOD'], i1)
                        self.hodpars = hod_funcs.HODParams(dic_hodpars, islogm0_0=True, islogm1_0=True)
                        hodprof = hod.HODProfile(cosmo, self.hodpars.lmminf, self.hodpars.sigmf, self.hodpars.fcf, self.hodpars.m0f, \
                                                     self.hodpars.m1f, self.hodpars.alphaf)
                        # Provide a, k grids
                        pk_hod_arr = np.log(np.array([hodprof.pk(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr]))
                        pk_hod = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=pk_hod_arr, is_logp=True)
                        logger.info('modHOD = {}. Using HOD to compute theory predictions.'.format(self.cl_params['modHOD']))
                        cls[:lmax_this+1] = ccl.angular_cl(cosmo, tracers[i1], tracers[i2], np.arange(lmax_this+1), p_of_k_a=pk_hod)
                    else:
                        logger.info('Only modHOD options zevol and bin supported.')
                        raise NotImplementedError()
                    #Extrapolate at high ell
                    cls[lmax_this+1:]=cls[lmax_this]*(cls[lmax_this]/cls[lmax_this-1])**(self.ells[lmax_this+1:]-lmax_this)

                    cls_conv = np.zeros(ndx.shape[0])
                    # Convolve with windows
                    for j in range(ndx.shape[0]):
                        cls_conv[j] = s.binning.windows[ndx[j]].convolve(cls)

                    if i1 == i2:
                        # We have an auto-correlation
                        if self.cl_params['fitNoise'] == 1:
                            cls_conv += params['Pw_s%i_bin%i'%(i, i1)]
                        else:
                            cls_conv += self.noise[i][i1]
                    cl_theory[i][ndx] = cls_conv

            # Add the theoretical cls to the context
            ctx.add('cl_theory', cl_theory)

        except:
            logging.warn('Runtime error caught from CCL. Used params [%s]'%( ', '.join([str(i) for i in p]) ) )
            raise LikelihoodComputationException()

    def get_params(self, params, paramtype, bin=None):

        params_subset = {}

        if paramtype == 'cosmo':
            KEYS = ['Omega_c', 'Omega_b', 'h', 'n_s', 'sigma8', 'A_s', 'Omega_k', 'Omega_g', 'Neff', 'm_nu',
                                'mnu_type', 'w0', 'wa', 'bcm_log10Mc', 'bcm_etab', 'bcm_ks', 'z_mg', 'df_mg',
                                'transfer_function', 'matter_power_spectrum', 'baryons_power_spectrum',
                                'mass_function', 'halo_concentration', 'emulator_neutrinos']
        elif paramtype == 'hod_zevol':
            KEYS = ['lmmin_0', 'lmmin_alpha', 'sigm_0', 'sigm_alpha', 'm0_0', 'm0_alpha', 'm1_0', 'm1_alpha', \
                  'alpha_0', 'alpha_alpha', 'fc_0', 'fc_alpha']
        elif paramtype == 'hod_bin':
            assert bin is not None, 'paramtype = {}, but bin number not given. Aborting.'.format(paramtype)
            BIN_KEYS = ['lmmin_0_bin{}'.format(bin), 'sigm_0_bin{}'.format(bin), 'm0_0_bin{}'.format(bin), \
                    'm1_0_bin{}'.format(bin), 'alpha_0_bin{}'.format(bin), 'fc_0_bin{}'.format(bin)]
            KEYS = ['lmmin_0', 'sigm_0', 'm0_0', 'm1_0', 'alpha_0', 'fc_0']
        else:
            return

        if paramtype != 'hod_bin':
            for key in KEYS:
                if key in params:
                    params_subset[key] = params[key]
        else:
            for i, key in enumerate(BIN_KEYS):
                if key in params:
                    params_subset[KEYS[i]] = params[key]

        return params_subset

    def get_tracers(self, sacc, cosmo, params):

        if 'z_b' in params:
            b_b = np.array([params['b_%2.1f'%z] for z in params['z_b']])

        tr_out = []
        for (tr_index, thistracer) in enumerate(sacc.tracers) :
            if thistracer.type.__contains__('point'):
                if 'b_bin' + str(tr_index) in params:
                    b_b = params['b_bin' + str(tr_index)]
                    z_b_arr = thistracer.z
                    b_b_arr = b_b*np.ones_like(z_b_arr)
                elif 'z_b' in params:
                    z_b = params['z_b']
                    bf = interp1d(z_b, b_b, kind='nearest') #Assuming linear interpolation. Decide on extrapolation.
                    z_b_arr = thistracer.z
                    b_b_arr = bf(z_b_arr) #Assuming that tracers have this attribute
                else:
                    raise ValueError("bias needed for each tracer")

                if 'zshift_bin' + str(tr_index) in params:
                    zbins = thistracer.z + params['zshift_bin' + str(tr_index)]
                else:
                    zbins = thistracer.z

                tr_out.append(ccl.NumberCountsTracer(cosmo, has_rsd=params['has_rsd'], dndz=(zbins, thistracer.Nz), \
                                                     bias=(z_b_arr, b_b_arr), mag_bias=params['has_magnification']))
            else :
                raise ValueError("Only \"point\" tracers supported")

        return tr_out

    def setup(self):
        """
        Sets up the core module.
        Tasks that need to be executed once per run
        """

        global hod_funcs
        if self.cl_params['modHOD'] == 'zevol':
            from desclss import hod_funcs
        elif self.cl_params['modHOD'] == 'bin':
            from desclss import hod_funcs_bin as hod_funcs

        # Provide a, k grids
        self.k_arr = np.logspace(-4.3, 1.5, 256)
        self.z_arr = np.linspace(0., 3., 50)[::-1]
        self.a_arr = 1./(1. + self.z_arr)

        if self.cl_params['modHOD'] != None and self.cl_params['fitHOD'] == 0:
            logger.info('Using HOD for theory predictions but not fitting parameters.')
            dic_hodpars = self.get_params(self.constants, 'hod_'+self.cl_params['modHOD'])
            self.hodpars = hod_funcs.HODParams(dic_hodpars, islogm0_0=True, islogm1_0=True)
