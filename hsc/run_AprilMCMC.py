#!/usr/bin/env python
import numpy as np
import sys,os
from Parameter import Parameter
from BaseLikelihood import BaseLikelihood
import MCMCAnalyzer
import argparse
import sacc
import pyccl as ccl
from hsc_like_mod import HSCLikeModule
from hsc_core_module import HSCCoreModule
from desclss.halo_mod_corr import HaloModCorrection
import yaml
import lk_max as lk
from cosmoHammer import ChainContext
from cosmoHammer.util import Params

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#######################################
# Initialize necessary parameter keys #
# Stolen from run_Cosmohammer         #
#######################################
Z_EFF = np.array([0.57, 0.70, 0.92, 1.25])
COSMO_PARAM_KEYS = ['Omega_b', 'Omega_k', 'sigma8', 'h', 'n_s', 'Omega_c']
BIAS_PARAM_BZ_KEYS = ['b_0.0', 'b_0.5', 'b_1.0', 'b_2.0', 'b_4.0']
z_b = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
BIAS_PARAM_CONST_KEYS = ['b_bin0', 'b_bin1', 'b_bin2', 'b_bin3']

##############################################################
# April_hsc_like module for simple MCMC sampling of hsc data #
##############################################################
class April_hsc_Like(BaseLikelihood):
    
    def __init__(self, HSC_CORE, HSC_LIKE, PARAMS):
        #print("call __init__")
        BaseLikelihood.__init__(self,"April_hsc_Like")
        self.hsc_core = HSC_CORE
        self.hsc_like = HSC_LIKE
        self.fit_params = PARAMS
        self.hsc_core.setup()
        self.hsc_like.setup()
        self.context = ChainContext( self.hsc_core.mapping , self.fit_params[:,0] )
        self.hsc_core(self.context)
    
    def freeParameters(self):
        #print("call freeParameters")
        Parameter_set = []
        for i, key in enumerate(self.hsc_core.mapping):
            # bounds ######### low ################## high #########
            limits = [ self.fit_params[i][1], self.fit_params[i][2] ]
            # parameter obj ############## name ###### value ############ error ############### bounds #
            Parameter_set.append( Parameter(key, self.fit_params[i][0], self.fit_params[i][3], limits) )
        return Parameter_set

    def updateParams(self,params):
        #print("call updateParameters")
        m = self.hsc_core.mapping
        for p in params:
            #print(p.name, p.value)
            self.fit_params[ m[p.name] ][0] = p.value
        self.context = ChainContext( self.hsc_core.mapping , self.fit_params[:,0] )
        self.hsc_core(self.context)
        #cl_theory = self.context.get('cl_theory')
        #print(self.fit_params[:,0])
 
    def loglike_wprior(self):
        #print("call loglike_wprior")
        return self.hsc_like.computeLikelihood(self.context)

###########################################
# Read in config file and it's parameters #
# Stolen from run_Cosmohammer             #
###########################################
parser = argparse.ArgumentParser(description='Calculate HSC clustering cls.')
parser.add_argument('--path2config', dest='path2config', type=str, help='Path to config file.', required=True)
parser.add_argument('--test-likelihood', dest='test_likelihood',  help='Prints a likelihood computation without fits', action='store_true')
args = parser.parse_args()
logger.info('Read args = {} from command line.'.format(args))
config = yaml.load(open(args.path2config))
logger.info('Read config from {}.'.format(args.path2config))
ch_config_params = config['ch_config_params']
saccs = [sacc.SACC.loadFromHDF(fn) for fn in config['saccfiles']]
logger.info ("Loaded {} sacc files.".format(len(saccs)))

if not os.path.isdir(ch_config_params['path2output']):
    try:
        os.makedirs(ch_config_params['path2output'])
        logger.info('Created directory {}.'.format(ch_config_params['path2output']))
    except:
        logger.info('Directory {} already exists.'.format(ch_config_params['path2output']))
        pass

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
            s.selectTracer(sacc_params['binNo'])
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
        s.selectTracer(sacc_params['binNo'])
        #print( len(s.tracers) )

Ntomo = len(saccs[0].tracers) ## number of tomo bins
logger.info ("Ntomo bins: %i"%Ntomo)

saccs, saccs_noise = lk.cutLranges(saccs, sacc_params['lmin'], sacc_params['lmax'], sacc_params['kmax'], Z_EFF, cosmo=None, Ntomo=Ntomo, logger=logger, saccs_noise=saccs_noise)

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
        
###################################
# April Likelihood setup and call #
###################################
L=April_hsc_Like(HSCCoreModule(param_mapping, config['default_params'], cl_params, saccs, noise, HMCorr=HMCorr), HSCLikeModule(saccs), params)
if args.test_likelihood:
    print("######################################")
    print("# Log Likelihood:", L.loglike_wprior(), "#")
    print("######################################")
    
else:
    sys.exit()
    if ch_config_params['use_mpi']==0:
        MCMCAnalyzer.MCMCAnalyzer(L,ch_config_params['path2output']+'/'+ch_config_params['chainsPrefix'], \
                                  ch_config_params['burninIterations'], ch_config_params['sampleIterations'])
    elif ch_config_params['use_mpi']==1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        print("hello world from rank:", rank)
        MCMCAnalyzer.MCMCAnalyzer(L,ch_config_params['path2output']+'/'+ch_config_params['chainsPrefix'], \
                                  ch_config_params['burninIterations'], ch_config_params['sampleIterations'], chain_num=(rank+1) )
    else:
        #p = L.freeParameters()
        #L.updateParams(p)
        print("Invalid MPI choice, choose (0 or 1)")