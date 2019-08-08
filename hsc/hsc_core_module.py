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

COSMO_PARAM_KEYS = ['Omega_b', 'Omega_k', 'sigma8', 'h', 'n_s', 'Omega_c']


class ClInterpolator(object):
    def __init__(self,lb,nrb=3,nb_dex_extrap_lo=10,kind='cubic'):
        """Interpolator for angular power spectra
        lb : central bandpower ells
        nrb : re-binning factor for ells within the range of the bandpowers
        nb_dex_extrap_lo : number of ells per decade for ells below the range of the bandpowers
        kind : interpolation type

        Extrapolation at high ell will be done assuming a power-law behaviour,
        with a power-law index estimated from the last two elements of the power spectrum.

        Once initialized, ClInterpolator.ls_eval holds the multipole values at which the
        power spectra should be estimated.
        """

        # Ells below the rannge
        ls_pre=np.geomspace(2, lb[0],nb_dex_extrap_lo*np.log10(lb[0]/2.))
        # Ells in range
        ls_mid=(lb[:-1,None]+(np.arange(nrb)[None,:]*np.diff(lb)[:,None]/nrb)).flatten()[1:]
        # Ells above range
        ls_post = np.geomspace(lb[-1], 2*lb[-1], 50)

        self.ls_eval = np.concatenate((ls_pre, ls_mid, ls_post))

        # Interpolation type
        self.kind = kind

    def interpolate_and_extrapolate(self,ls,clb):
        """Go from a C_ell estimated in a few ells to one estimated in a
        finer grid of ells.

        ls : finer grid of ells
        clb : power spectra evaluated at self.ls_eval

        returns : power spectrum evaluated at ls
        """

        # Ells in range
        ind_good = np.where(ls<=self.ls_eval[-1])[0]
        ind_bad = np.where(ls>self.ls_eval[-1])[0]
        clret = np.zeros(len(ls))
        cli = interp1d(self.ls_eval,clb,kind=self.kind,fill_value=0,bounds_error=False)
        clret[ind_good] = cli(ls[ind_good])

        # Extrapolate at high ell
        clret[ind_bad] = clb[-1]*(ls[ind_bad]/self.ls_eval[-1])**-1.05

        return clret

class HSCCoreModule(object):
    """
    Core Module for calculating HSC clustering cls.
    """

    def __init__(self, PARAM_MAPPING, DEFAULT_PARAMS, cl_params, saccs, noise, HMCorr=None):
        """
        Constructor of the HSCCoreModule
        """

        self.mapping = PARAM_MAPPING
        self.constants = DEFAULT_PARAMS
        self.cl_params = cl_params
        self.saccs = saccs
        self.noise = noise
        self.lmax = self.saccs[0].binning.windows[0].w.shape[0]
        self.ells = np.arange(self.lmax)

        if set(COSMO_PARAM_KEYS) <= set(DEFAULT_PARAMS.keys()):
            FID_COSMO_PARAMS = self.get_params(DEFAULT_PARAMS, 'cosmo')
            logger.info('Not varying cosmological parameters. Fixing cosmology to fiducial values = {}.'.format(FID_COSMO_PARAMS))
            self.cosmo = ccl.Cosmology(**FID_COSMO_PARAMS)

        if HMCorr is not None:
            logger.info('HMCorr provided. Correcting halo model.')
            self.corr_halo_mod = True
            self.HMCorr = HMCorr
        else:
            logger.info('Not correcting halo model.')
            self.corr_halo_mod = False
            self.HMCorr = None

        if 'magBias' not in self.cl_params or self.cl_params['magBias'] == 0:
            logger.info('magBias = 0. Not including magnification bias in theory predictions.')
            self.mag_bias = False
        elif self.cl_params['magBias'] == 1:
            logger.info('magBias = 1. Including magnification bias in theory predictions.')
            assert 'path2sfunc' in self.cl_params, 'magBias set to 1 but no tabulated s function provided. Aborting.'
            self.mag_bias = True
            z, s = np.genfromtxt(self.cl_params['path2sfunc'], unpack=True)
            self.mag_bias_z = z
            self.mag_bias_s = s
            global mag_bias
            import estimate_mag_bias as mag_bias
        else:
            raise NotImplementedError('Only values supported for magBias parameter are magBias = 0, 1. Aborting.')


    def __call__(self, ctx):
        """
        Compute theoretical prediction for clustering power spectra and store in the context.
        """
        # Get the parameters from the context
        p = ctx.getParams()

        params = self.constants.copy()
        for k,v in self.mapping.items():
            params[k] = p[v]

        cl_theory = [np.zeros((s.size(),)) for s in self.saccs]

        cosmo_params = self.get_params(params, 'cosmo')

        try:
            if (cosmo_params.viewkeys() & self.mapping.viewkeys()) != set([]):
                cosmo = ccl.Cosmology(**cosmo_params)
            else:
                cosmo = self.cosmo

            for i, s in enumerate(self.saccs):
                tracers = self.get_tracers(s, cosmo, params, self.cl_params)

                if self.cl_params['fitHOD'] == 1 and self.cl_params['modHOD'] == 'zevol':
                    dic_hodpars = self.get_params(params, 'hod_'+self.cl_params['modHOD'])
                    self.hodpars = hod_funcs.HODParams(dic_hodpars, islogm0=True, islogm1=True)

                if self.cl_params['modHOD'] == 'zevol':
                    hodprof = hod.HODProfile(cosmo, self.hodpars.lmminf, self.hodpars.sigmf, self.hodpars.fcf, self.hodpars.m0f, \
                                                 self.hodpars.m1f, self.hodpars.alphaf)
                    # Compute HOD Pk
                    if not self.mag_bias:
                        pk_hod_arr = np.array([hodprof.pk(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr])
                        # pk_hod_arr = np.array([hodprof.pk(self.k_arr, a) for a in self.a_arr])
                    else:
                        # pk_mm_arr = np.array([ccl.halomodel.halomodel_matter_power(cosmo, self.k_arr, a) for a in self.a_arr])
                        pk_mm_arr = np.array([ccl.nonlin_matter_power(cosmo, self.k_arr, a) for a in self.a_arr])
                        pk_gg_arr = np.array([hodprof.pk(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr])
                        pk_gm_arr = np.array([hodprof.pk_gm(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr])

                    # Correct halo model Pk
                    if self.corr_halo_mod:
                        if not self.mag_bias:
                            pk_hod_arr *= self.rk_hm
                        else:
                            # pk_mm_arr *= self.rk_hm
                            pk_gg_arr *= self.rk_hm
                            pk_gm_arr *= self.rk_hm

                    if not self.mag_bias:
                        pk_hod_arr = np.log(pk_hod_arr)
                        pk_hod = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=pk_hod_arr, is_logp=True)
                    else:
                        pk_mm = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=np.log(pk_mm_arr), is_logp=True)
                        pk_gg = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=np.log(pk_gg_arr), is_logp=True)
                        pk_gm = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=np.log(pk_gm_arr), is_logp=True)

                for i1, i2, _, ells_binned, ndx in s.sortTracers() :
                    itp = ClInterpolator(ells_binned)
                    if self.cl_params['modHOD'] is None:
                        logger.info('modHOD = {}. Not using HOD to compute theory predictions.'.format(self.cl_params['modHOD']))
                        clb = ccl.angular_cl(cosmo, tracers[i1], tracers[i2], itp.ls_eval)

                    elif self.cl_params['modHOD'] == 'zevol':
                        logger.info('modHOD = {}. Using HOD to compute theory predictions.'.format(self.cl_params['modHOD']))
                        if not self.mag_bias:
                            logger.info('Not including magnification bias in theory predictions.')
                            clb = ccl.angular_cl(cosmo, tracers[i1], tracers[i2], itp.ls_eval, p_of_k_a=pk_hod)
                        else:
                            logger.info('Including magnification bias in theory predictions.')
                            clb = mag_bias.angular_cl(cosmo, tracers[i1], tracers[i2], itp.ls_eval, pk_mm, pk_gg, pk_gm)

                    elif self.cl_params['modHOD'] == 'bin':
                        dic_hodpars = self.get_params(params, 'hod_'+self.cl_params['modHOD'], i1)
                        self.hodpars = hod_funcs.HODParams(dic_hodpars, islogm0=True, islogm1=True)
                        hodprof = hod.HODProfile(cosmo, self.hodpars.lmminf, self.hodpars.sigmf, self.hodpars.fcf, self.hodpars.m0f, \
                                                     self.hodpars.m1f, self.hodpars.alphaf)
                        # Compute HOD Pk
                        if not self.mag_bias:
                            pk_hod_arr = np.array([hodprof.pk(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr])
                            # pk_hod_arr = np.array([hodprof.pk(self.k_arr, a) for a in self.a_arr])
                        else:
                            # pk_mm_arr = np.array([ccl.halomodel.halomodel_matter_power(cosmo, self.k_arr, a) for a in self.a_arr])
                            pk_mm_arr = np.array([ccl.nonlin_matter_power(cosmo, self.k_arr, a) for a in self.a_arr])
                            pk_gg_arr = np.array([hodprof.pk(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr])
                            pk_gm_arr = np.array([hodprof.pk_gm(self.k_arr, a, lmmin=8., lmmax=16., nlm=128) for a in self.a_arr])

                        # Correct halo model Pk
                        if self.corr_halo_mod:
                            if not self.mag_bias:
                                pk_hod_arr *= self.rk_hm
                            else:
                                # pk_mm_arr *= self.rk_hm
                                pk_gg_arr *= self.rk_hm
                                pk_gm_arr *= self.rk_hm

                        if not self.mag_bias:
                            pk_hod_arr = np.log(pk_hod_arr)
                            pk_hod = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=pk_hod_arr, is_logp=True)
                        else:
                            pk_mm = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=np.log(pk_mm_arr), is_logp=True)
                            pk_gg = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=np.log(pk_gg_arr), is_logp=True)
                            pk_gm = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=np.log(pk_gm_arr), is_logp=True)

                        logger.info('modHOD = {}. Using HOD to compute theory predictions.'.format(self.cl_params['modHOD']))
                        if not self.mag_bias:
                            logger.info('Not including magnification bias in theory predictions.')
                            clb = ccl.angular_cl(cosmo, tracers[i1], tracers[i2], itp.ls_eval, p_of_k_a=pk_hod)
                        else:
                            logger.info('Including magnification bias in theory predictions.')
                            clb = mag_bias.angular_cl(cosmo, tracers[i1], tracers[i2], itp.ls_eval, pk_mm, pk_gg, pk_gm)

                    else:
                        logger.info('Only modHOD options zevol and bin supported.')
                        raise NotImplementedError()
                    # Extrapolate at high ell
                    cls = itp.interpolate_and_extrapolate(self.ells, clb)
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
            KEYS = ['lmmin_0', 'lmmin_1', 'sigm_0', 'sigm_1', 'm0_0', 'm0_1', 'm1_0', 'm1_1', \
                  'alpha_0', 'alpha_1', 'fc_0', 'fc_1', 'lmmin', 'lmminp', 'm0', 'm0p', 'm1', 'm1p', 'zfid']
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

    def get_tracers(self, sacc, cosmo, params, cl_params):

        if 'z_b' in params:
            b_b = np.array([params['b_%2.1f'%z] for z in params['z_b']])

        tr_out = []
        for (tr_index, thistracer) in enumerate(sacc.tracers) :
            if thistracer.type.__contains__('point'):
                if 'b_bin{}'.format(tr_index) in params:
                    b_b = params['b_bin{}'.format(tr_index)]
                    z_b_arr = thistracer.z
                    b_b_arr = b_b*np.ones_like(z_b_arr)
                elif 'z_b' in params:
                    z_b = params['z_b']
                    bf = interp1d(z_b, b_b, kind='nearest') #Assuming linear interpolation. Decide on extrapolation.
                    z_b_arr = thistracer.z
                    b_b_arr = bf(z_b_arr) #Assuming that tracers have this attribute
                else:
                    raise ValueError("bias needed for each tracer")

                if 'zshift_bin{}'.format(tr_index) in params:
                    zbins = thistracer.z + params['zshift_bin{}'.format(tr_index)]

                else:
                    zbins = thistracer.z

                if 'pzMethod' in cl_params:
                    if cl_params['pzMethod'] != 'COSMOS30':
                        Nz = thistracer.extra_cols[cl_params['pzMethod']]
                    else:
                        Nz = thistracer.Nz
                else:
                    Nz = thistracer.Nz

                if self.mag_bias:
                    if 'zshift_bin{}'.format(tr_index) in params:
                        mb_z = self.mag_bias_z + params['zshift_bin{}'.format(tr_index)]
                    else:
                        mb_z = self.mag_bias_z

                    mb_s = self.mag_bias_s

                if not self.mag_bias:
                    tr_out.append(ccl.NumberCountsTracer(cosmo, has_rsd=params['has_rsd'], dndz=(zbins[zbins>=0.], Nz[zbins>=0.]), \
                                                        bias=(z_b_arr, b_b_arr), mag_bias=params['has_magnification']))
                else:
                    # Setup the galaxy tracer
                    # In generalized tracers the bias is treated as a transfer function in a
                    # a needs to be increasing
                    a_b_arr = 1./(1. + z_b_arr[::-1])
                    b_b_arr = b_b_arr[::-1]

                    g_tracer = ccl.Tracer()
                    g_kernel = ccl.get_density_kernel(cosmo, (zbins[zbins>=0.], Nz[zbins>=0.]))
                    g_tracer.add_tracer(cosmo=cosmo, kernel=g_kernel, transfer_a=(a_b_arr, b_b_arr))
                    # Setup magnification bias tracer
                    m_tracer = ccl.Tracer()
                    chis, w = ccl.get_lensing_kernel(cosmo, (zbins[zbins>=0.], Nz[zbins>=0.]), (mb_z[mb_z>=0.], mb_s[mb_z>=0.]))
                    if 'mb_ampl' in params:
                        m_kernel = (chis, -2.*params['mb_ampl']*w)
                    else:
                        m_kernel = (chis, -2.*w)
                    m_tracer.add_tracer(cosmo, kernel=m_kernel, der_bessel=-1, der_angles=1)

                    tr_out.append([g_tracer, m_tracer])
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
            from desclss import hod_funcs_evol_fit as hod_funcs
        elif self.cl_params['modHOD'] == 'bin':
            from desclss import hod_funcs_bin as hod_funcs

        # Provide a, k grids
        # self.k_arr = np.logspace(-4.3, 3, 1000)
        self.k_arr = np.logspace(-4.3, 1.5, 256)
        self.z_arr = np.linspace(0., 3., 50)[::-1]
        self.a_arr = 1./(1. + self.z_arr)

        if self.cl_params['modHOD'] != None and self.cl_params['fitHOD'] == 0:
            logger.info('Using HOD for theory predictions but not fitting parameters.')
            dic_hodpars = self.get_params(self.constants, 'hod_'+self.cl_params['modHOD'])
            self.hodpars = hod_funcs.HODParams(dic_hodpars, islogm0=True, islogm1=True)

        if self.corr_halo_mod:
            logger.info('Correcting halo model Pk with HALOFIT ratio.')
            self.rk_hm = self.HMCorr.rk_interp(self.k_arr, self.a_arr)
