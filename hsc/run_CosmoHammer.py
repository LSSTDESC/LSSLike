#! /usr/bin/env python

# I haven't made any real changes just want to access on multiple computers

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import os
import argparse
import sacc
import pyccl as ccl
from cosmoHammer import LikelihoodComputationChain
from hsc_like_mod import HSCLikeModule
from hsc_core_module import HSCCoreModule
from InitializeFromChain import InitializeFromChain
from desclss.halo_mod_corr import HaloModCorrection
import yaml

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Z_EFF = np.array([0.57, 0.70, 0.92, 1.25])

COSMO_PARAM_KEYS = ['Omega_b', 'Omega_k', 'sigma8', 'h', 'n_s', 'Omega_c']

BIAS_PARAM_BZ_KEYS = ['b_0.0', 'b_0.5', 'b_1.0', 'b_2.0', 'b_4.0']
z_b = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
BIAS_PARAM_CONST_KEYS = ['b_bin0', 'b_bin1', 'b_bin2', 'b_bin3']

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

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', required=True)

args = parser.parse_args()

logger.info('Read args = {} from command line.'.format(args))

config = yaml.load(open(args.path2config))
logger.info('Read config from {}.'.format(args.path2config))

ch_config_params = config['ch_config_params']

logger.info('Called hsc_driver with saccfiles = {}.'.format(config['saccfiles']))
saccs = [sacc.SACC.loadFromHDF(fn) for fn in config['saccfiles']]
logger.info ("Loaded {} sacc files.".format(len(saccs)))

# Make path to output
if not os.path.isdir(ch_config_params['path2output']):
    try:
        os.makedirs(ch_config_params['path2output'])
        logger.info('Created directory {}.'.format(ch_config_params['path2output']))
    except:
        logger.info('Directory {} already exists.'.format(ch_config_params['path2output']))
        pass

if ch_config_params['use_mpi'] == 0:
    from cosmoHammer import CosmoHammerSampler
else:
    from cosmoHammer import MpiCosmoHammerSampler

cl_params = config['cl_params']
sacc_params = config['sacc_params']

# Determine noise from noise saccs
if cl_params['fitNoise'] == 0:
    logger.info('Not fitting shot noise. Determining from noise sacc.')

    try:
        fnames_saccs_noise = [os.path.splitext(fn)[0]+'_noise.sacc' for fn in config['saccfiles']]
        logger.info('Reading noise saccs {}.'.format(fnames_saccs_noise))
        saccs_noise = [sacc.SACC.loadFromHDF(fn) for fn in fnames_saccs_noise]
        logger.info ("Loaded %i noise sacc files."%len(saccs))
    except:
        fnames_saccs_noise = [os.path.join(os.path.split(fn)[0], 'noi_bias.sacc') for fn in config['saccfiles']]
        logger.info('Reading noise saccs {}.'.format(fnames_saccs_noise))
        saccs_noise = [sacc.SACC.loadFromHDF(fn) for fn in fnames_saccs_noise]
        logger.info ("Loaded %i noise sacc files."%len(saccs))

    # Add precision matrix to noise saccs
    for i, s in enumerate(saccs):
        saccs_noise[i].precision = s.precision

    if sacc_params['joinSaccs'] == 1:
        saccs_noise = [sacc.coadd(saccs_noise, mode='area')]
    if sacc_params['cullCross'] == 1:
        for s in saccs_noise:
            s.cullCross()
    if sacc_params['singleBin'] == 1:
        assert sacc_params['binNo'] is not None, 'Single bin fit requested but bin number not specified. Aborting.'
        for s in saccs_noise:
            s.selectTracer(args.binNo)
else:
    saccs_noise = None

if sacc_params['joinSaccs'] == 1:
    saccs=[sacc.coadd(saccs, mode='area')]
if sacc_params['cullCross'] == 1:
    for s in saccs:
        s.cullCross()
if sacc_params['singleBin'] == 1:
    assert sacc_params['binNo'] is not None, 'Single bin fit requested but bin number not specified. Aborting.'
    for s in saccs:
        s.selectTracer(args.binNo)

Ntomo = len(saccs[0].tracers) ## number of tomo bins
logger.info ("Ntomo bins: %i"%Ntomo)

saccs, saccs_noise = cutLranges(saccs, sacc_params['lmin'], sacc_params['lmax'], sacc_params['kmax'], Z_EFF, cosmo=None, Ntomo=Ntomo, saccs_noise=saccs_noise)

noise = [[0 for i in range(Ntomo)] for ii in range(len(saccs_noise))]
for i, s in enumerate(saccs_noise):
    for ii in range(Ntomo):
        binmask = (s.binning.binar['T1']==ii)&(s.binning.binar['T2']==ii)
        noise[i][ii] = s.mean.vector[binmask]

if cl_params['corrHM'] == 1:
    assert cl_params['modHOD'] is not None, 'Halo model correction requested but not using HOD for theory predictions. Aborting.'

    if set(COSMO_PARAM_KEYS) <= set(config['default_params']):
        FID_COSMO_PARAMS = {}
        for key in COSMO_PARAM_KEYS:
            FID_COSMO_PARAMS[key] = config['default_params'][key]
    else:
        assert 'fid_cosmo_params' in cl_params, 'Halo model correction requested but no fiducial cosmological model provided. Aborting.'
        FID_COSMO_PARAMS = config['fid_cosmo_params']

    logger.info('Setting up halo model correction with fixed cosmological parameters set to {}.'.format(FID_COSMO_PARAMS))
    cosmo = ccl.Cosmology(**FID_COSMO_PARAMS)
    HMCorr = HaloModCorrection(cosmo, k_range=[1e-4, 1e2], nlk=256, z_range=[0., 3.], nz=50)
else:
    HMCorr = None
    FID_COSMO_PARAMS = None

fit_params = config['fit_params']
param_mapping = {}
nparams = len(fit_params.keys())
params = np.zeros((nparams, 4))
for key in fit_params.keys():
    param_mapping[key] = fit_params[key][0]
    params[fit_params[key][0], :] = fit_params[key][1:]

if set(BIAS_PARAM_BZ_KEYS) <= set(param_mapping.keys()):
    logger.info('Fitting for galaxy bias with model = bz.')
    config['default_params']['z_b'] = z_b

elif set(BIAS_PARAM_CONST_KEYS) <= set(param_mapping.keys()):
    logger.info('Fitting for galaxy bias with model = const.')

else:
    assert 'bg' in cl_params, 'Not fitting for galaxy bias but no fiducial values provided. Aborting.'
    logger.info('Not fitting for galaxy bias.')
    config['default_params'].update(cl_params['bg'])
    if set(BIAS_PARAM_BZ_KEYS) <= set(cl_params['bg'].keys()):
        logger.info('Galaxy bias model = bz.')
        config['default_params']['z_b'] = z_b

    else:
        logger.info('Galaxy bias model = const.')

# Set up CosmoHammer
chain = LikelihoodComputationChain(
                    min=params[:, 1],
                    max=params[:, 2])

chain.addCoreModule(HSCCoreModule(param_mapping, config['default_params'], cl_params, saccs, noise, HMCorr=HMCorr))

chain.addLikelihoodModule(HSCLikeModule(saccs))

chain.setup()

if ch_config_params['use_mpi'] == 0:
    if ch_config_params['rerun'] == 0:
        sampler = CosmoHammerSampler(
                    params= params,
                    likelihoodComputationChain=chain,
                    filePrefix=os.path.join(ch_config_params['path2output'], ch_config_params['chainsPrefix']),
                    walkersRatio=ch_config_params['walkersRatio'],
                    burninIterations=ch_config_params['burninIterations'],
                    sampleIterations=ch_config_params['sampleIterations'])
    else:
        assert ch_config_params['path2rerunchain'] is not None, 'rerun is {}, but path to rerun chains not set. Aborting.'.format(ch_config_params['rerun'])
        path2chain = args.path2rerunchain
        sampler = CosmoHammerSampler(
                    params= params,
                    likelihoodComputationChain=chain,
                    filePrefix=os.path.join(ch_config_params['path2output'], ch_config_params['chainsPrefix']),
                    walkersRatio=ch_config_params['walkersRatio'],
                    burninIterations=ch_config_params['burninIterations'],
                    sampleIterations=ch_config_params['sampleIterations'],
                    initPositionGenerator=InitializeFromChain(ch_config_params['path2rerunchain'], fraction = 0.8))
else:
    if ch_config_params['rerun'] == 0:
        sampler = MpiCosmoHammerSampler(
                    params= params,
                    likelihoodComputationChain=chain,
                    filePrefix=os.path.join(ch_config_params['path2output'], ch_config_params['chainsPrefix']),
                    walkersRatio=ch_config_params['walkersRatio'],
                    burninIterations=ch_config_params['burninIterations'],
                    sampleIterations=ch_config_params['sampleIterations'])
    else:
        assert ch_config_params['path2rerunchain'] is not None, 'rerun is {}, but path to rerun chains not set. Aborting.'.format(ch_config_params['rerun'])
        sampler = MpiCosmoHammerSampler(
                    params= params,
                    likelihoodComputationChain=chain,
                    filePrefix=os.path.join(ch_config_params['path2output'], ch_config_params['chainsPrefix']),
                    walkersRatio=ch_config_params['walkersRatio'],
                    burninIterations=ch_config_params['burninIterations'],
                    sampleIterations=ch_config_params['sampleIterations'],
                    initPositionGenerator=InitializeFromChain(ch_config_params['path2rerunchain'], fraction = 0.8))

sampler.startSampling()

