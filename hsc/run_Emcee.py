#!/usr/bin/python3

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import os
import argparse
import sacc
import pyccl as ccl
from hsc_like_mod import HSCLikeModule
from hsc_core_module import HSCCoreModule
from InitializeFromChain import InitializeFromChain
from desclss.halo_mod_corr import HaloModCorrection
import yaml
import time
import sys
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Z_EFF = np.array([0.57, 0.70, 0.92, 1.25])

COSMO_PARAM_KEYS = ['Omega_b', 'Omega_k', 'sigma8', 'h', 'n_s', 'Omega_c']

BIAS_PARAM_BZ_KEYS = ['b_0.0', 'b_0.5', 'b_1.0', 'b_2.0', 'b_4.0']
z_b = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
BIAS_PARAM_CONST_KEYS = ['b_bin0', 'b_bin1', 'b_bin2', 'b_bin3']


class SampleFileUtil(object):
    """Util for handling sample files.
    Copied from Andrina's code.

    :param filePrefix: the prefix to use
    :param reuseBurnin: True if the burn in data from a previous run should be used
    """
    def __init__(self, filePrefix, carry_on=False):
        self.filePrefix = filePrefix
        if carry_on:
            mode = 'a'
        else:
            mode = 'w'
        self.samplesFile = open(self.filePrefix + '.txt', mode)
        self.probFile = open(self.filePrefix + 'prob.txt', mode)

    def persistSamplingValues(self, pos, prob):
        self.persistValues(self.samplesFile, self.probFile, pos, prob)

    def persistValues(self, posFile, probFile, pos, prob):
        """Writes the walker positions and the likelihood to the disk
        """
        posFile.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
        posFile.write("\n")
        posFile.flush()

        probFile.write("\n".join([str(p) for p in prob]))
        probFile.write("\n")
        probFile.flush();

    def close(self):
        self.samplesFile.close()
        self.probFile.close()

    def __str__(self, *args, **kwargs):
        return "SampleFileUtil"


def cutLranges(saccs, kmax, cosmo, Ntomo, zeff=None, saccs_noise=None):
    logger.info('zeff not provided. Computing directly from sacc.')
    zeff = np.zeros(Ntomo)
    for i, t in enumerate(saccs[0].tracers):
        zeff[i] = t.meanZ()
    logger.info('zeff = {}.'.format(zeff))

    assert Ntomo == zeff.shape[0], 'zeff shape does not match number of tomographic bins.'
    logger.info('Computing lmax according to specified kmax = {}.'.format(kmax))

    lmax = kmax2lmax(kmax, zeff, cosmo)

    if Ntomo == 1:
        lmin = [0]
    elif Ntomo == 4:
        lmin=[0,0,0,0]
    else:
        print ("weird Ntomo")

    logger.info('lmin = {}, lmax = {}.'.format(lmin, lmax))

    for i, s in enumerate(saccs):
        s.cullLminLmax(lmin, lmax)
        if saccs_noise is not None:
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
        cosmo = ccl.Cosmology(n_s=0.9649, A_s=2.1e-9, h=0.6736, Omega_c=0.264, Omega_b=0.0493,
                              transfer_function='boltzmann_class')

    # Comoving angular diameter distance in Mpc
    chi_A = ccl.comoving_angular_distance(cosmo, 1./(1.+zeff))
    lmax = kmax*chi_A - 1./2.

    return lmax

parser = argparse.ArgumentParser(description='Calculate HSC clustering cls.')

parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', required=True)
parser.add_argument('--time-likelihood', dest='time_likelihood',  help='Times the likelihood calculations', action='store_true')

# Read command-line arguments
args = parser.parse_args()
logger.info('Read args = {} from command line.'.format(args))

# Read params from yaml
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

#if ch_config_params['use_mpi'] == 0:
#    from cosmoHammer import CosmoHammerSampler
#else:
#    from cosmoHammer import MpiCosmoHammerSampler

cl_params = config['cl_params']
sacc_params = config['sacc_params']

# Determine noise from noise saccs
logger.info('Not fitting shot noise. Determining from noise sacc.')

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
    assert 'binNo' in sacc_params.keys(), 'Single bin fit requested but bin number not specified. Aborting.'
    logger.info('Only fitting bin = {}.'.format(sacc_params['binNo']))
    for s in saccs_noise:
        s.selectTracer(sacc_params['binNo'])
else:
    if 'fitBins' in sacc_params.keys():
        logger.info('Only fitting bins = {}.'.format(sacc_params['fitBins']))
        for s in saccs_noise:
            s.selectTracers(sacc_params['fitBins'])

if sacc_params['joinSaccs'] == 1:
    saccs=[sacc.coadd(saccs, mode='area')]
if sacc_params['cullCross'] == 1:
    for s in saccs:
        s.cullCross()
if sacc_params['singleBin'] == 1:
    assert 'binNo' in sacc_params.keys(), 'Single bin fit requested but bin number not specified. Aborting.'
    logger.info('Only fitting bin = {}.'.format(sacc_params['binNo']))
    for s in saccs:
        s.selectTracer(sacc_params['binNo'])
else:
    if 'fitBins' in sacc_params.keys():
        logger.info('Only fitting bins = {}.'.format(sacc_params['fitBins']))
        for s in saccs:
            s.selectTracers(sacc_params['fitBins'])

Ntomo = len(saccs[0].tracers) ## number of tomo bins
logger.info ("Ntomo bins: %i"%Ntomo)

saccs, saccs_noise = cutLranges(saccs, sacc_params['kmax'], cosmo=None,
                                Ntomo=Ntomo, saccs_noise=saccs_noise)

noise = [[0 for i in range(Ntomo)] for ii in range(len(saccs_noise))]
for i, s in enumerate(saccs_noise):
    for ii in range(Ntomo):
        binmask = (s.binning.binar['T1']==ii)&(s.binning.binar['T2']==ii)
        noise[i][ii] = s.mean.vector[binmask]

if 'path2cov' in sacc_params.keys():
    logger.info('Covariance matrix provided. Setting precision matrix of saccs and saccs_noise to provided covariance matrix.')
    for i in range(len(saccs)):
        covmat = np.load(sacc_params['path2cov'][i])
        logger.info('Read covariance matrix from {}.'.format(sacc_params['path2cov'][i]))
        saccs[i].precision = sacc.Precision(covmat, 'dense', is_covariance=True)
        saccs_noise[i].precision = sacc.Precision(covmat, 'dense', is_covariance=True)

if cl_params['corrHM']:
    assert cl_params['modHOD'] is not None, 'Halo model correction requested but not using HOD for theory predictions. Aborting.'

    if set(COSMO_PARAM_KEYS) <= set(config['default_params']):
        FID_COSMO_PARAMS = {}
        for key in COSMO_PARAM_KEYS:
            FID_COSMO_PARAMS[key] = config['default_params'][key]
    else:
        assert 'fid_cosmo_params' in cl_params, 'Halo model correction requested but no fiducial cosmological model provided. Aborting.'
        FID_COSMO_PARAMS = cl_params['fid_cosmo_params']

    logger.info('Setting up halo model correction with fixed cosmological parameters set to {}.'.format(FID_COSMO_PARAMS))
    cosmo = ccl.Cosmology(transfer_function='boltzmann_class', **FID_COSMO_PARAMS)
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

logger.info('Not fitting for galaxy bias.')
config['default_params'].update(cl_params['bg'])

if 'temperature' in ch_config_params.keys():
    logger.info('temperature = {}. Supplying likelihood with temperature.'.format(ch_config_params['temperature']))
    temperature = ch_config_params['temperature']

else:
    logger.info('No temperature provided. Supplying likelihood with no temperature.')
    temperature = None

th = HSCCoreModule(param_mapping, config['default_params'], cl_params, saccs, noise, HMCorr=HMCorr)
th.setup()
lik = HSCLikeModule(saccs, temperature)
lik.setup()

def inrange(p):
    return np.all((p<=params[:, 2]) & (p>=params[:, 1]))

def lnprob(p):
    if inrange(p):
        try:
            cl_theory = th.compute_theory(p)
            lnP = lik.computeLikelihoodFromCl(cl_theory)
        except:
            lnP = -np.inf
    else:
        lnP = -np.inf
    return lnP


if args.time_likelihood:
    print('   ==========================================')
    print("   | Calculating likelihood evaluation time |")
    print('   ==========================================')  
    timing = np.zeros(10)
    for i in range(-1,10):
        if i==-1:
            print('Burn test')
        else:
            print('Test ',i,' of 9')
        start = time.time()
        if i<5:
            lnprob(params[:, 0]+i*0.1*params[:, 3])
        else:
            lnprob(params[:, 0]-(i-4)*0.2*params[:, 3])
        finish = time.time()
        if i>-1:
            timing[i] = finish-start

    mean2 = np.mean(timing)
    print('============================================================================')
    print('mean computation time: ', mean2)
    stdev = np.std(timing)
    print('standard deviation : ', stdev)
    print('============================================================================')
else:
    import emcee
    class DumPool(object):
        def __init__(self):
            pass

        def is_master(self):
            return True

        def close(self):
            pass

    if ch_config_params['use_mpi']:
        from schwimmbad import MPIPool
        pool = MPIPool()
        print("Using MPI")
        pool_use = pool
    else:
        pool = DumPool()
        print("Not using MPI")
        pool_use = None

    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    nwalkers = nparams * ch_config_params['walkersRatio']
    nsteps = ch_config_params['burninIterations'] + ch_config_params['sampleIterations']

    prefix_chain = os.path.join(ch_config_params['path2output'],
                                ch_config_params['chainsPrefix'])
    found_file = os.path.isfile(prefix_chain+'.txt')

    if (not found_file) or (not ch_config_params['rerun']):
        p_initial = params[:, 0] + np.random.normal(size=(nwalkers, nparams)) * params[:, 3][None, :]
        nsteps_use = nsteps
    else:
        print("Restarting from a previous run")
        old_chain = np.loadtxt(prefix_chain+'.txt')
        p_initial = old_chains[-nwalkers:,:]
        nsteps_use = max(nsteps-len(old_chain) // nwalkers, 0)

    chain_file = SampleFileUtil(prefix_chain, carry_on=ch_config_params['rerun'])
    sampler = emcee.EnsembleSampler(nwalkers, nparams, lnprob, pool=pool_use)
    start = time.time()
    print("Running %d samples" % nsteps_use)
    count = 1
    for pos, prob, _ in sampler.sample(p_initial, iterations=nsteps_use):
        if pool.is_master():
            print('Iteration done. Persisting.')
            chain_file.persistSamplingValues(pos, prob)

            if counter % 10:
                print(f"Finished sample {counter}")
        counter += 1

    pool.close()
    print("Took ",(end - start)," seconds")
