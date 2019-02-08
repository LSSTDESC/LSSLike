#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import time
import os
import argparse
import sacc
import pyccl as ccl
from cosmoHammer import LikelihoodComputationChain
from hsc_like_mod import HSCLikeModule
from hsc_core_module import HSCCoreModule


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOD_PARAM_KEYS = ['lmmin_0', 'lmmin_alpha', 'sigm_0', 'sigm_alpha', 'm0_0', 'm0_alpha', 'm1_0', 'm1_alpha', \
                  'alpha_0', 'alpha_alpha', 'fc_0', 'fc_alpha']
HOD_PARAM_MEANS = np.atleast_2d(np.array([10., 0., 0.35, 0.3, 7.5, 0.7, 13., 0.1, 1., 0.3, 0.25, -1.5]))
HOD_PARAM_WIDTHS = np.atleast_2d(np.array([1., 0.1, 0.1, 0.1, 1., 0.1, 1., 0.1, 0.1, 0.1, 0.1, 0.1]))
HOD_PARAM_MINS = np.atleast_2d(np.array([9., -1., 0., -1., 5.5, -1., 11., -1., 0., -1., 0., -3.]))
HOD_PARAM_MAXS = np.atleast_2d(np.array([15., 1., 0.8, 1., 13., 1., 17., 1., 2., 1., 1., 0.]))

HOD_BIN_PARAM_KEYS = []
for i in range(4):
    HOD_BIN_PARAM_KEYS += ['lmmin_0_bin{}'.format(i), 'sigm_0_bin{}'.format(i), 'm0_0_bin{}'.format(i), \
                           'm1_0_bin{}'.format(i), 'alpha_0_bin{}'.format(i), 'fc_0_bin{}'.format(i)]
HOD_BIN_PARAM_MEANS = np.atleast_2d(np.tile(np.array([10., 0.35, 7.5, 13., 1., 0.25]), 4))
HOD_BIN_PARAM_WIDTHS = np.atleast_2d(np.tile(np.array([1., 0.1, 1., 1., 0.1, 0.1]), 4))
HOD_BIN_PARAM_MINS = np.atleast_2d(np.tile(np.array([9., 0., 5.5, 11., 0., 0.]), 4))
HOD_BIN_PARAM_MAXS = np.atleast_2d(np.tile(np.array([15., 0.8, 13., 17., 2., 1.]), 4))

HOD_SINGLE_BIN_PARAM_KEYS = ['lmmin_0_bin0', 'sigm_0_bin0', 'm0_0_bin0', 'm1_0_bin0', 'alpha_0_bin0', 'fc_0_bin0']
HOD_SINGLE_BIN_PARAM_MEANS = np.atleast_2d(np.array([10., 0.35, 7.5, 13., 1., 0.25]))
HOD_SINGLE_BIN_PARAM_WIDTHS = np.atleast_2d(np.array([1., 0.1, 1., 1., 0.1, 0.1]))
HOD_SINGLE_BIN_PARAM_MINS = np.atleast_2d(np.array([9., 0., 5.5, 11., 0., 0.]))
HOD_SINGLE_BIN_PARAM_MAXS = np.atleast_2d(np.array([15., 0.8, 13., 17., 2., 1.]))

BIAS_PARAM_BZ_KEYS = ['b_0.0', 'b_0.5', 'b_1.0', 'b_2.0', 'b_4.0']
BIAS_PARAM_BZ_MEANS = np.atleast_2d(np.array([0.7, 1.5, 1.8, 2.0, 2.5]))
BIAS_PARAM_BZ_WIDTHS = np.atleast_2d(0.1*np.ones(len(BIAS_PARAM_BZ_KEYS)))
BIAS_PARAM_BZ_MINS = np.atleast_2d(0.5*np.ones(len(BIAS_PARAM_BZ_KEYS)))
BIAS_PARAM_BZ_MAXS = np.atleast_2d(5.*np.ones(len(BIAS_PARAM_BZ_KEYS)))

BIAS_PARAM_CONST_KEYS = ['b_bin0', 'b_bin1', 'b_bin2', 'b_bin3']
BIAS_PARAM_CONST_MEANS = np.atleast_2d(np.array([0.7, 1.5, 1.8, 2.0]))
BIAS_PARAM_CONST_WIDTHS = np.atleast_2d(0.1*np.ones(len(BIAS_PARAM_CONST_KEYS)))
BIAS_PARAM_CONST_MINS = np.atleast_2d(0.5*np.ones(len(BIAS_PARAM_CONST_KEYS)))
BIAS_PARAM_CONST_MAXS = np.atleast_2d(5.*np.ones(len(BIAS_PARAM_CONST_KEYS)))

Z_EFF = np.array([0.57, 0.70, 0.92, 1.25])

def cutLranges(saccs, lmin, lmax, kmax, zeff, cosmo, Ntomo, saccs_noise=None):
    # lmax as a function of sample
    if lmax=='auto':
        if Ntomo==1:
            lmax=[1000]
            lmin=[0]
        elif Ntomo==4:
            lmax=[1000,2000,3000,4000]
            lmin=[0,0,0,0]
        else:
            print ("weird Ntomo")

    elif lmax == 'kmax':
        assert kmax is not None, 'kmax not provided.'
        assert zeff is not None, 'zeff array not provided.'
        assert Ntomo == zeff.shape[0], 'zeff shape does not match number of tomographic bins.'
        logger.info('Computing lmax according to specified kmax = {}.'.format(kmax))

        lmax = kmax2lmax(kmax, zeff, cosmo)

        if Ntomo == 1:
            lmin = [0]
        elif Ntomo == 4:
            lmin=[0,0,0,0]
        else:
            print ("weird Ntomo")

    else:
        lmin=lmin
        lmax=lmax

    logger.info('lmin = {}, lmax = {}.'.format(lmin, lmax))

    if saccs_noise is None:
        for s in saccs:
            s.cullLminLmax(lmin, lmax)
    # If saccs_noise is not None, also cull those
    else:
        for i, s in enumerate(saccs):
            s.cullLminLmax(lmin, lmax)
            saccs_noise[i].cullLminLmax(lmin, lmax)

    return saccs, saccs_noise

def kmax2lmax(kmax, zeff, cosmo=None):
    """
    Determine lmax corresponding to given kmax at an effective redshift zeff according to
    kmax = (lmax + 1/2)/chi(zeff)
    :param kmax: maximal wavevector in Mpc^-1
    :param zeff: effective redshift of sample
    :return lmax: maximal angular multipole corresponding to kmax
    """

    if cosmo is None:
        logger.info('CCL cosmology object not supplied. Initializing with Planck 2018 cosmological parameters.')
        cosmo = ccl.Cosmology(n_s=0.9649, A_s=2.1e-9, h=0.6736, Omega_c=0.264, Omega_b=0.0493)

    # Comoving angular diameter distance in Mpc
    chi_A = ccl.comoving_angular_distance(cosmo, 1./(1.+zeff))
    lmax = kmax*chi_A - 1./2.

    return lmax

parser = argparse.ArgumentParser(description='Calculate HSC clustering cls.')

parser.add_argument('--path2output', dest='path2output', type=str, help='Path to output.', required=True)
parser.add_argument('--chainsPrefix', dest='chainsPrefix', type=str, help='Prefix of output chains.', required=True)
parser.add_argument('--fitBias', dest='fitBias', type=int, help='Tag denoting if to fit for bias parameters.', required=False, default=1)
parser.add_argument('--biasMod', dest='biasMod', type=str, help='Tag denoting which bias model to use. biasMod = {bz, const}.', required=False, default='bz')
parser.add_argument('--fitNoise', dest='fitNoise', type=int, help='Tag denoting if to fit shot noise.', required=False, default=0)
parser.add_argument('--lmin', dest='lmin', type=str, help='Tag specifying how lmin is determined. lmin = {auto, kmax}.', required=False, default='auto')
parser.add_argument('--lmax', dest='lmax', type=str, help='Tag specifying how lmax is determined. lmax = {auto, kmax}.', required=False, default='auto')
parser.add_argument('--kmax', dest='kmax', type=float, help='If lmax=kmax, this sets kmax to use.', required=False)
parser.add_argument('--modHOD', dest='modHOD', type=str, help='Tag denoting which HOD model to use in theory predictions.', required=False)
parser.add_argument('--fitHOD', dest='fitHOD', type=int, help='Tag denoting if to fit for HOD parameters.', required=False, default=0)
parser.add_argument('--joinSaccs', dest='joinSaccs', type=int, help='Option to join sacc files into one.', required=False, default=1)
parser.add_argument('--cullCross', dest='cullCross', type=int, help='Option to remove all cross-correlations from fit.', required=False, default=1)
parser.add_argument('--singleBin', dest='singleBin', type=int, help='Option to fit only one redshift bin.', required=False, default=0)
parser.add_argument('--binNo', dest='binNo', type=int, help='Index of redshift bin to fit.', required=False)
parser.add_argument('--platfrm', dest='platfrm', type=str, help='Platform where code is being run, options = {local, cluster}.', required=False, default='local')
parser.add_argument('--fixCosmo', dest='fixCosmo', type=int, help='Tag denoting if to fix cosmological parameters.', required=False, default=0)
parser.add_argument('--saccfiles', dest='saccfiles', nargs='+', help='Path to saccfiles.', required=True)

args = parser.parse_args()

logger.info('Called hsc_driver with saccfiles = {}.'.format(args.saccfiles))
saccs = [sacc.SACC.loadFromHDF(fn) for fn in args.saccfiles]
logger.info ("Loaded {} sacc files.".format(len(saccs)))

# Make path to output
if not os.path.isdir(args.path2output):
    try:
        os.makedirs(args.path2output)
        logger.info('Created directory {}.'.format(args.path2output))
    except:
        logger.info('Directory {} already exists.'.format(args.path2output))
        pass

if args.platfrm == 'local':
    from cosmoHammer import CosmoHammerSampler
else:
    from cosmoHammer import MpiCosmoHammerSampler

cl_params = {'fitHOD': args.fitHOD,
             'modHOD': args.modHOD,
             'fitNoise': args.fitNoise}

# Determine noise from noise saccs
if args.fitNoise == 0:
    logger.info('Not fitting shot noise. Determining from noise sacc.')

    fnames_saccs_noise = [os.path.splitext(fn)[0]+'_noise.sacc' for fn in args.saccfiles]
    logger.info('Reading noise saccs {}.'.format(fnames_saccs_noise))

    # New filenames
    # fnames_saccs_noise = [os.path.join(os.path.split(fn)[0], 'noi_bias.sacc') for fn in args.saccfiles]
    # logger.info('Reading noise saccs {}.'.format(fnames_saccs_noise))
    try:
        saccs_noise = [sacc.SACC.loadFromHDF(fn) for fn in fnames_saccs_noise]
        logger.info ("Loaded %i noise sacc files."%len(saccs))
    except IOError:
        raise IOError("Need to provide noise saccs.")

    # Add precision matrix to noise saccs
    for i, s in enumerate(saccs):
        saccs_noise[i].precision = s.precision

    if args.joinSaccs == 1:
        saccs_noise = [sacc.coadd(saccs_noise, mode='area')]
    if args.cullCross == 1:
        for s in saccs_noise:
            s.cullCross()
    if args.singleBin == 1:
        assert args.binNo is not None, 'Single bin fit requested but bin number not specified. Aborting.'
        for s in saccs_noise:
            s.selectTracer(args.binNo)
else:
    saccs_noise = None

if args.joinSaccs == 1:
    saccs=[sacc.coadd(saccs, mode='area')]
if args.cullCross == 1:
    for s in saccs:
        s.cullCross()
if args.singleBin == 1:
    assert args.binNo is not None, 'Single bin fit requested but bin number not specified. Aborting.'
    for s in saccs:
        s.selectTracer(args.binNo)

Ntomo = len(saccs[0].tracers) ## number of tomo bins
logger.info ("Ntomo bins: %i"%Ntomo)

saccs, saccs_noise = cutLranges(saccs, args.lmin, args.lmax, args.kmax, Z_EFF, cosmo=None, Ntomo=Ntomo, saccs_noise=saccs_noise)

noise = [[0 for i in range(Ntomo)] for ii in range(len(saccs_noise))]
for i, s in enumerate(saccs_noise):
    for ii in range(Ntomo):
        binmask = (s.binning.binar['T1']==ii)&(s.binning.binar['T2']==ii)
        noise[i][ii] = s.mean.vector[binmask]

# Default paramters
DEFAULT_PARAMS = {'Omega_b': 0.0486,
                 'Omega_k': 0.0,
                 'Omega_nu': 0.001436176,
                 'sigma8': 0.8,
                 'h': 0.6774,
                 'n_s': 0.96,
                 'transfer_function': 'boltzmann_class',
                 'matter_power_spectrum': 'halofit',
                 'has_rsd': False,
                 'has_magnification': None
                }

if args.fixCosmo == 1:
    FID_COSMO_PARAMS = {'Omega_b': 0.0486,
                     'Omega_k': 0.0,
                     'sigma8': 0.8,
                     'h': 0.6774,
                     'n_s': 0.96,
                     'Omega_c': 0.25,
                     'transfer_function': 'boltzmann_class',
                     'matter_power_spectrum': 'halofit'
                    }
    logger.info('Fixing comsological parameters to {}.'.format(FID_COSMO_PARAMS))
    params = np.array([])
    PARAM_MAPPING = {}

else:
    logger.info('Not fixing cosmological parameters.')
    # Parameter start center, min, max, start width
    params = np.array([[0.25, 0.1, 0.5, 0.01]])

    # Cosmological parameters and mapping we will fit
    PARAM_MAPPING = {'Omega_c': 0
                    }

if args.fitBias == 1:
    if args.biasMod == 'bz':
        logger.info('Fitting galaxy bias with bias model = {}.'.format(args.biasMod))
        PARAM_MAPPING.update(dict(zip(BIAS_PARAM_BZ_KEYS, np.arange(len(PARAM_MAPPING), \
                                                len(PARAM_MAPPING) + len(BIAS_PARAM_BZ_KEYS), dtype='int'))))
        tempparams = np.concatenate((BIAS_PARAM_BZ_MEANS, BIAS_PARAM_BZ_MINS, BIAS_PARAM_BZ_MAXS, \
                                     BIAS_PARAM_BZ_WIDTHS), axis=0)
        if params.shape[0] != 0:
            params = np.vstack((params, tempparams.T))
        else:
            params = tempparams.T

        z_b = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        DEFAULT_PARAMS['z_b'] = z_b

    elif args.biasMod == 'const':
        logger.info('Fitting galaxy bias with bias model = {}.'.format(args.biasMod))
        PARAM_MAPPING.update(dict(zip(BIAS_PARAM_CONST_KEYS, np.arange(len(PARAM_MAPPING), \
                                                len(PARAM_MAPPING) + len(BIAS_PARAM_CONST_KEYS), dtype='int'))))
        tempparams = np.concatenate((BIAS_PARAM_CONST_MEANS, BIAS_PARAM_CONST_MINS, BIAS_PARAM_CONST_MAXS, \
                                     BIAS_PARAM_CONST_WIDTHS), axis=0)
        if params.shape[0] != 0:
            params = np.vstack((params, tempparams.T))
        else:
            params = tempparams.T

else:
    if args.biasMod == 'bz':
        logger.info('Not fitting galaxy bias with bias model = {}.'.format(args.biasMod))
        logger.info('Setting galaxy bias to 1.')
        z_b = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        DEFAULT_PARAMS['z_b'] = z_b

        b_bz_keys = ['b_0.0', 'b_0.5', 'b_1.0', 'b_2.0', 'b_4.0']
        b_bz = np.ones(5)
        for i, bin in enumerate(b_bz_keys):
            DEFAULT_PARAMS[bin] = b_bz[i]

    elif args.biasMod == 'const':
        logger.info('Not fitting galaxy bias with bias model = {}.'.format(args.biasMod))
        logger.info('Setting galaxy bias to 1.')
        b_bin_keys = ['b_bin0', 'b_bin1', 'b_bin2', 'b_bin3']
        b_bin = np.ones(4)
        for i, bin in enumerate(b_bin_keys):
            DEFAULT_PARAMS[bin] = b_bin[i]

if args.fitHOD == 1:
    if args.modHOD == 'zevol':
        PARAM_MAPPING.update(dict(zip(HOD_PARAM_KEYS, np.arange(len(PARAM_MAPPING), \
                                        len(PARAM_MAPPING) + len(HOD_PARAM_KEYS), dtype='int'))))
        tempparams = np.concatenate((HOD_PARAM_MEANS, HOD_PARAM_MINS, HOD_PARAM_MAXS, \
                                         HOD_PARAM_WIDTHS), axis=0)
    elif args.modHOD == 'bin':
        if args.singleBin == 0:
            PARAM_MAPPING.update(dict(zip(HOD_BIN_PARAM_KEYS, np.arange(len(PARAM_MAPPING), \
                                            len(PARAM_MAPPING) + len(HOD_BIN_PARAM_KEYS), dtype='int'))))
            tempparams = np.concatenate((HOD_BIN_PARAM_MEANS, HOD_BIN_PARAM_MINS, HOD_BIN_PARAM_MAXS, \
                                             HOD_BIN_PARAM_WIDTHS), axis=0)
        else:
            PARAM_MAPPING.update(dict(zip(HOD_SINGLE_BIN_PARAM_KEYS, np.arange(len(PARAM_MAPPING), \
                                            len(PARAM_MAPPING) + len(HOD_SINGLE_BIN_PARAM_KEYS), dtype='int'))))
            tempparams = np.concatenate((HOD_SINGLE_BIN_PARAM_MEANS, HOD_SINGLE_BIN_PARAM_MINS, HOD_SINGLE_BIN_PARAM_MAXS, \
                                             HOD_SINGLE_BIN_PARAM_WIDTHS), axis=0)

    else:
        logger.info('Only modHOD options bin or z_evol supported. Aborting.')
        raise NotImplementedError()

    if params.shape[0] != 0:
        params = np.vstack((params, tempparams.T))
    else:
        params = tempparams.T

# Set up CosmoHammer
chain = LikelihoodComputationChain(
                    min=params[:,1],
                    max=params[:,2])

if args.fixCosmo == 1:
    chain.addCoreModule(HSCCoreModule(PARAM_MAPPING, DEFAULT_PARAMS, cl_params, saccs, noise, \
                                      fid_cosmo_params=FID_COSMO_PARAMS))
else:
    chain.addCoreModule(HSCCoreModule(PARAM_MAPPING, DEFAULT_PARAMS, cl_params, saccs, noise))

chain.addLikelihoodModule(HSCLikeModule(saccs))

chain.setup()

if args.platfrm == 'local':
    sampler = CosmoHammerSampler(
                params= params,
                likelihoodComputationChain=chain,
                filePrefix=os.path.join(args.path2output, args.chainsPrefix),
                walkersRatio=2,
                burninIterations=3000,
                sampleIterations=1000)
else:
    sampler = MpiCosmoHammerSampler(
            params= params,
            likelihoodComputationChain=chain,
            filePrefix=os.path.join(args.path2output, args.chainsPrefix),
            walkersRatio=8,
            burninIterations=3000,
            sampleIterations=1000)

start = time.time()
sampler.startSampling()

