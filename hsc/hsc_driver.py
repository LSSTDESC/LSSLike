#!/usr/bin/env python
import sys
import numpy as np
import logging
import scipy.optimize
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pyccl as ccl
import argparse
import os

import sacc
from desclss import LSSTheory,LSSLikelihood

import emcee as mc
from ParamVec import ParamVec

HOD_PARAM_KEYS = ['lmmin_0', 'lmmin_1', 'sigm_0', 'sigm_1', 'm0_0', 'm0_1', 'm1_0', 'm1_1', \
                  'alpha_0', 'alpha_1', 'fc_0', 'fc_1']
HOD_PARAM_MINS = np.array([9., -1., 0., -1., 10**6.5, -1., 10**11.5, -1., 0., -1., 0., -2.])
HOD_PARAM_MAXS = np.array([15., 1., 0.8, 1., 10**13., 1., 10**17., 1., 2., 1., 1., 0.])

DEFAULTLIKEEXC = -1e9

class HSCAnalyze:

    def __init__(self, fnames, lmin='auto', lmax='auto', kmax=None, zeff=None, cosmo=None,
                 fitOc=True, Oc=0.25,
                 fits8=False, s8=0.8,
                 fith0=False, h0 = 0.6774,
                 fitBias=True, BiasMod='bz',
                 zbias=[0.0,0.5,1.0,2.0,4.0],
                 bias=[0.7,1.5,1.8,2.0,2.5],
                 fitNoise=True, noise=0.75, ## in units of 1e-8
                 fitPZShifts=False,
                 pzshifts=[0,0,0,0],
                 join_saccs=True,   ## If true, Cinverse add all saccs into one
                 cull_cross=True,  ## If true, use just auto
                 log=logging.DEBUG,
                 hod=0,
                 fitHOD=0,
                 hodpars=None, corr_halo_mod=True):

        assert join_saccs, 'Shot noise currently only supports coadded saccs.'

        if type(log)==logging.Logger:
            self.log=log
        else:
            self.log = logging.getLogger('HSCAnalyze')
            self.log.setLevel(log)
            ch = logging.StreamHandler()
            ch.setLevel(log)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)
            self.log.propagate = False

        self.log.info('Called hsc_driver with saccfiles = {}.'.format(fnames))
        self.saccs=[sacc.SACC.loadFromHDF(fn) for fn in fnames]
        self.log.info ("Loaded %i sacc files."%len(self.saccs))

        # Determine noise from noise saccs
        if noise is None:
            self.log.info('No shot noise parameter provided. Determining from noise sacc.')

            # self.log.info('Reading noise saccs {}.'.format(fnames_saccs_noise))
            try:
                fnames_saccs_noise = [os.path.splitext(fn)[0]+'_noise.sacc' for fn in fnames]
                self.saccs_noise = [sacc.SACC.loadFromHDF(fn) for fn in fnames_saccs_noise]
                self.log.info ("Loaded %i noise sacc files."%len(self.saccs))
            except:
                fnames_saccs_noise = [os.path.split(fn)[0]+'/noi_bias.sacc' for fn in fnames]
                self.saccs_noise = [sacc.SACC.loadFromHDF(fn) for fn in fnames_saccs_noise]
                self.log.info ("Loaded %i noise sacc files."%len(self.saccs))
                # raise IOError("Noise = {}, need to provide noise saccs.".format(noise))

            # Add precision matrix to noise saccs
            for i, s in enumerate(self.saccs):
                self.saccs_noise[i].precision = s.precision

            if join_saccs:
                self.saccs_noise = [sacc.coadd(self.saccs_noise, mode='area')]
            if cull_cross:
                for s in self.saccs_noise:
                    s.cullCross()

        else:
            self.saccs_noise = None

        if join_saccs:
            self.saccs=[sacc.coadd(self.saccs, mode='area')]
        if cull_cross:
            for s in self.saccs:
                s.cullCross()

        self.Ntomo=len(self.saccs[0].tracers) ## number of tomo bins
        self.log.info ("Ntomo bins: %i"%self.Ntomo)

        self.fixnames()
        self.cutLranges(lmin, lmax, kmax, zeff, cosmo)

        if not (type(noise)==list):
            if noise is not None:
                self.log.info('Scalar shot noise parameter provided. Setting constant for all tomographic bins.')
                noise=[noise]*self.Ntomo
            else:
                noise = [0 for i in range(self.Ntomo*len(self.saccs_noise))]
                for s in self.saccs_noise:
                    for i in range(self.Ntomo):
                        binmask = (s.binning.binar['T1']==i)&(s.binning.binar['T2']==i)
                        noise[i] = s.mean.vector[binmask]*1e8

                # noise = [(1./np.mean(t.extra_cols['ndens']))*1e8 for t in self.saccs[0].tracers]
        else:
            noise = noise

        self.log.info('Shot noise array = {}.'.format(noise))

        assert len(noise) == len(self.saccs)*self.Ntomo, 'Noise list shape does not match total number of tracers.'
        assert len(pzshifts) == self.Ntomo, 'pzshifts array shape does not match number of tomographic bins.'

        self.setParametes(fitOc,Oc,fits8,s8,fith0,h0,fitBias,BiasMod,zbias,bias,fitNoise,noise,fitPZShifts,pzshifts, \
                          hod, fitHOD, hodpars)
            
        self.lts=[LSSTheory(s, hod=hod, fitHOD=fitHOD, hodpars=hodpars, corr_halo_mod=corr_halo_mod) for s in self.saccs]
        self.lks=[LSSLikelihood(s) for s in self.saccs]

        self.dofs = self.ncls - len(self.P)
        self.log.info('dofs = {}.'.format(self.dofs))

        
    def fixnames(self):
        
        ## let's rename exp_sample, because into hscgals, because these are the same gals
        ## and at the same time also add a "name" field in case we need it later
        for s in self.saccs:
            assert (len(s.tracers)==len(self.saccs[0].tracers)) ## Let's require the same number of bins
            for t in s.tracers:
                t.name=t.exp_sample
                t.exp_sample='hscgals'


    def cutLranges(self, lmin, lmax, kmax, zeff, cosmo):
        # lmax as a function of sample
        if lmax=='auto':
            if self.Ntomo==1:
                ## distance to 0.6 = 1500 Mpc/h
                ## lmax = 0.3Mpc/h*1500 so 450??
                self.lmax=[1000]
                self.lmin=[0]
            elif self.Ntomo==4:
                self.lmax=[1000,2000,3000,4000]
                self.lmin=[0,0,0,0]
            else:
                print ("weird Ntomo")
                stop()
        elif lmax == 'kmax':
            assert kmax is not None, 'kmax not provided.'
            assert zeff is not None, 'zeff array not provided.'
            assert self.Ntomo == zeff.shape[0], 'zeff shape does not match number of tomographic bins.'
            self.log.info('Computing lmax according to specified kmax = {}.'.format(kmax))

            self.lmax = self.kmax2lmax(kmax, zeff, cosmo)

            if self.Ntomo == 1:
                self.lmin = [0]
            elif self.Ntomo == 4:
                self.lmin=[0,0,0,0]
            else:
                print ("weird Ntomo")
                stop()

        else:
            self.lmin=lmin
            self.lmax=lmax

        self.log.info('lmin = {}, lmax = {}.'.format(self.lmin, self.lmax))

        self.ncls = 0
        if self.saccs_noise is None:
            for s in self.saccs:
                s.cullLminLmax(self.lmin,self.lmax)
                self.ncls += s.mean.vector.shape[0]
        # If saccs_noise is not None, also cull those
        else:
            for i, s in enumerate(self.saccs):
                s.cullLminLmax(self.lmin,self.lmax)
                self.saccs_noise[i].cullLminLmax(self.lmin,self.lmax)
                self.ncls += s.mean.vector.shape[0]

        self.log.info('ncls = {}.'.format(self.ncls))

    def kmax2lmax(self, kmax, zeff, cosmo=None):
        """
        Determine lmax corresponding to given kmax at an effective redshift zeff according to
        kmax = (lmax + 1/2)/chi(zeff)
        :param kmax: maximal wavevector in Mpc^-1
        :param zeff: effective redshift of sample
        :return lmax: maximal angular multipole corresponding to kmax
        """

        if cosmo is None:
            self.log.info('CCL cosmology object not supplied. Initializing with Planck 2018 cosmological parameters.')
            cosmo = ccl.Cosmology(n_s=0.9649, A_s=2.1e-9, h=0.6736, Omega_c=0.264, Omega_b=0.0493)

        # Comoving angular diameter distance in Mpc
        chi_A = ccl.comoving_angular_distance(cosmo, 1./(1.+zeff))
        lmax = kmax*chi_A - 1./2.

        return lmax

    def setParametes(self,fitOc,Oc,fits8,s8,fith0,h0,fitBias,BiasMod,zbias,bias,fitNoise,noise,fitPZShifts,pzshifts, hod, \
                     fitHOD, hodpars):
        #### set up parameters
        self.fitOc=fitOc
        self.Oc=Oc
        self.fits8=fits8
        self.s8=s8
        self.h0=h0
        self.fith0=fith0
        self.fitBias=fitBias
        self.BiasMod=BiasMod
        self.zbias=zbias
        self.bias=bias
        self.fitNoise=fitNoise
        self.noise=noise
        self.fitPZShifts=fitPZShifts
        self.pzshifts=pzshifts
        # HOD
        self.hod = hod
        self.fitHOD = fitHOD
        if self.hod == 1:
            self.log.info('hod = {}. Computing theory predictions with HOD.'.format(self.hod))
            self.hodpars = {}
            for i, key in enumerate(HOD_PARAM_KEYS):
                self.hodpars[key] = hodpars[i]
            
        self.P=ParamVec()
        if self.fitOc:
            self.P.addParam('Oc',Oc,'$\\Omega_c$',min=0.2,max=0.4)
        if self.fits8:
            self.P.addParam('s8',s8, '$\sigma_8$',min=0.1,max=2.0)
        if self.fith0:
            self.P.addParam('h0',h0, '$h_0$',min=0.65,max=0.75)

        if self.fitBias:
            if self.BiasMod == 'bz':
                for z,b in zip(self.zbias,self.bias):
                    self.P.addParam('b_%2.1f'%z,b,min=0.5,max=5)
            elif self.BiasMod == 'const':
                for i, b in enumerate(self.bias):
                    self.P.addParam('b_bin%i'%i, b, min=0.5, max=5.)
            else:
                raise ValueError("Initial value for bias needed.")

        if self.fitNoise:
            for i in range(self.Ntomo):
                self.P.addParam('Pw_%i'%i,noise[i],min=0.0,max=10)  ## in units of 1e-8

        if self.fitPZShifts:
            for i in range(self.Ntomo):
                self.P.addParam('s_%i'%i,0.0,min=-0.5,max=+0.5)
            
        if self.fitHOD == 1:
            self.log.info('Fitting HOD parameters.')
            for i, key in enumerate(HOD_PARAM_KEYS):
                self.P.addParam('{}'.format(key), hodpars[i], min=HOD_PARAM_MINS[i], max=HOD_PARAM_MAXS[i])

        self.log.info ("Parameters: "+str(self.P._names))


    def predictTheory(self,p):
        P=self.P.clone()
        P.setValues(p)
        oc=P.value('Oc') if self.fitOc else self.Oc
        s8=P.value('s8') if self.fits8 else self.s8
        h0=P.value('h0') if self.fith0 else self.h0

        dic={'Omega_c':oc,
             'Omega_b': 0.0493,
             'Omega_k': 0.0,
             'sigma8': 0.8111,
             'h': 0.6736,
             'n_s': 0.9649,
             'transfer_function': 'boltzmann_class',
             'matter_power_spectrum': 'halofit',
             'has_rsd':False,
             'has_magnification':None,
             'mass_function': 'tinker10',
             # 'pzMethod': 'nz_frankenz'
            }

        if self.fitBias:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':[P.value('b_%2.1f'%z) for z in self.zbias]})
            elif self.BiasMod == 'const':
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = P.value('b_bin%i'%i)
            else:
                raise ValueError('BiasMod needs to be set and bias parameters need to be provided.')
        else:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':self.bias})
            elif self.BiasMod == 'const':
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = self.bias[i]
            else:
                raise ValueError('BiasMod needs to be set and bias parameters need to be provided.')
        
        if self.fitPZShifts:
            for i in range(self.Ntomo):
                dic['zshift_bin%i'%i]=P.value('s_%i'%i)

        if self.hod == 1:
            if self.fitHOD == 1:
                for i, key in enumerate(HOD_PARAM_KEYS):
                    dic['{}'.format(key)]= P.value('{}'.format(key))
            else:
                for i, key in enumerate(HOD_PARAM_KEYS):
                    dic['{}'.format(key)]= self.hodpars['{}'.format(key)]

        cls=[0 for lt in self.lts]

        for i, lt in enumerate(self.lts):
            if self.fitNoise:
                for ii in range(self.Ntomo):
                    dic['Pw_bin%i'%ii]=P.value('Pw_%i'%ii)*1e-8
            else:
                for ii in range(self.Ntomo):
                    dic['Pw_bin%i'%ii]=self.noise[ii]*1e-8
            cls[i] = lt.get_prediction(dic)

        return cls

    def predictNoSNTheory(self,p):
        P=self.P.clone()
        P.setValues(p)
        oc=P.value('Oc') if self.fitOc else self.Oc
        s8=P.value('s8') if self.fits8 else self.s8
        h0=P.value('h0') if self.fith0 else self.h0

        dic={'Omega_c':oc,
             'Omega_b': 0.0493,
             'Omega_k': 0.0,
             'sigma8': 0.8111,
             'h': 0.6736,
             'n_s': 0.9649,
             'transfer_function': 'boltzmann_class',
             'matter_power_spectrum': 'halofit',
             'has_rsd':False,
             'has_magnification':None,
             'mass_function': 'tinker10',
             # 'pzMethod': 'nz_frankenz'
            }

        if self.fitBias:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':[P.value('b_%2.1f'%z) for z in self.zbias]})
            elif self.BiasMod == 'const':
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = P.value('b_bin%i'%i)
            else:
                raise ValueError('BiasMod needs to be set and bias parameters need to be provided.')
        else:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':self.bias})
            elif self.BiasMod == 'const':
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = self.bias[i]
            else:
                raise ValueError('BiasMod needs to be set and bias parameters need to be provided.')

        if self.fitPZShifts:
            for i in range(self.Ntomo):
                dic['zshift_bin%i'%i]=P.value('s_%i'%i)

        if self.hod == 1:
            if self.fitHOD == 1:
                for i, key in enumerate(HOD_PARAM_KEYS):
                    dic['{}'.format(key)]= P.value('{}'.format(key))
            else:
                for i, key in enumerate(HOD_PARAM_KEYS):
                    dic['{}'.format(key)]= self.hodpars['{}'.format(key)]

        cls=[lt.get_noSN_prediction(dic) for lt in self.lts]

        return cls

    def predict1h_2hTheory(self,p):
        P=self.P.clone()
        P.setValues(p)
        oc=P.value('Oc') if self.fitOc else self.Oc
        s8=P.value('s8') if self.fits8 else self.s8
        h0=P.value('h0') if self.fith0 else self.h0

        dic={'Omega_c':oc,
             'Omega_b': 0.0493,
             'Omega_k': 0.0,
             'sigma8': 0.8111,
             'h': 0.6736,
             'n_s': 0.9649,
             'transfer_function': 'boltzmann_class',
             'matter_power_spectrum': 'halofit',
             'has_rsd':False,
             'has_magnification':None,
             'mass_function': 'tinker10',
             # 'pzMethod': 'nz_frankenz'
            }

        if self.fitBias:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':[P.value('b_%2.1f'%z) for z in self.zbias]})
            elif self.BiasMod == 'const':
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = P.value('b_bin%i'%i)
            else:
                raise ValueError('BiasMod needs to be set and bias parameters need to be provided.')
        else:
            if self.BiasMod == 'bz':
                dic.update({'hscgals_z_b':self.zbias,
                            'hscgals_b_b':self.bias})
            elif self.BiasMod == 'const':
                for i in range(self.Ntomo):
                    dic['hscgals_b_bin%i'%i] = self.bias[i]
            else:
                raise ValueError('BiasMod needs to be set and bias parameters need to be provided.')

        if self.fitPZShifts:
            for i in range(self.Ntomo):
                dic['zshift_bin%i'%i]=P.value('s_%i'%i)

        if self.hod == 1:
            if self.fitHOD == 1:
                for i, key in enumerate(HOD_PARAM_KEYS):
                    dic['{}'.format(key)]= P.value('{}'.format(key))
            else:
                for i, key in enumerate(HOD_PARAM_KEYS):
                    dic['{}'.format(key)]= self.hodpars['{}'.format(key)]

        cls_1h = [0 for lt in self.lts]
        cls_2h = [0 for lt in self.lts]
        b = [0 for lt in self.lts]
        for i, lt in enumerate(self.lts):
            if self.fitNoise:
                for ii in range(self.Ntomo):
                    dic['Pw_bin%i'%ii]=P.value('Pw_%i'%ii)*1e-8
            else:
                for ii in range(self.Ntomo):
                    dic['Pw_bin%i'%ii]=self.noise[ii]*1e-8
            cls_1h_temp, cls_2h_temp, b_temp = lt.get_1h_2h_prediction(dic)
            cls_1h[i] = cls_1h_temp
            cls_2h[i] = cls_2h_temp
            b[i] = b_temp

        return cls_1h, cls_2h, b
    
    #Define log(p). This is just a wrapper around the LSSLikelihood lk
    def logprobs(self,p):

        try:
            cls=self.predictTheory(p)
            #print (cls)
            likes=np.array([lk(cl) for lk,cl in zip(self.lks,cls)])
            # dof = np.array([len(cl) for cl in cls])
        except:
            self.log.warning("Caught error from CCL. Setting likelihoods to DEFAULTLIKEEXC.")
            likes = DEFAULTLIKEEXC*np.ones(len(self.lks))

        self.chisq_cur = -2*likes.sum()
        self.log.debug("parameters: "+str(p)+" -> chi2= "+str(self.chisq_cur)+" dof = ncls - nparam: "+str(self.dofs))

        return likes

    def logprob(self,p):
        return self.logprobs(p).sum()
    
    def plotDataTheory(self, params=None, path2fig=None):
        """
        Plots the theory predictions with and without shot noise.
        :param params:
        :param path2fig:
        :return:
        """

        P = self.P.clone()
        if params is not None:
            P.setValues(params)

        fig = plt.figure()
        subplot=fig.add_subplot(111)
        self.log.info('Best-fit parameters = {}.'.format(P.values()))
        cls=self.predictTheory(P.values())

        cls_sn_rem = self.predictNoSNTheory(P.values())

        for i,s in enumerate(self.saccs):
            #plt.subplot(3,3,i+1)
            self.plot_vector(s, subplot,plot_corr = 'auto',prediction=cls[i],weightpow=0,
                          label=self.saccs[0].tracers[0].name, show_axislabels = True, show_legend=True, linestyle_pred=':')
            self.plot_vector(s, subplot,plot_corr = 'auto',prediction=cls_sn_rem[i],weightpow=0,
                          label=self.saccs[0].tracers[0].name, show_axislabels = True, show_legend=False, linestyle_pred='--')
            try:
                subplot.text(0.05, 0.15, r'$\chi^2/ \mathrm{{dof}} = {:.2f}$'.format(self.chisq_cur/self.dofs)+'\n'+r'$\mathrm{{dof}} = {}$'.format(self.dofs), \
                         transform=subplot.transAxes, fontsize=14, verticalalignment='top')
            except:
                self.log.warning('Not writing chi2 in plot. Chi2 not set.')
            #plt.title(s.tracers[0].name)
        if path2fig is not None:
            plt.savefig(path2fig)
        plt.show()


    def plot_vector (self, sacc, subplot = None, plot_corr='all', weightpow = 2, set_logx=True, set_logy=True,
                    show_axislabels = False, show_legend=True, prediction=None, label=None, linestyle_pred=':'):
        """
        Plots the mean vector associated to the different tracers
        in the sacc file. The tracer correlations to plot can be selected by
        passing a list of pairs of values in plot_corr.  It can also plot
        the autocorrelation by passing 'auto', the cross-correlation by
        passng 'cross', and both by passing 'all'.  The C_ell's will be
        weighted by a factor of ell^{weightpow}.
        """

        if subplot is None:
            fig = plt.figure()
            subplot = fig.add_subplot(111)

        if sacc.precision is not None:
            errs=np.sqrt(sacc.precision.getCovarianceMatrix().diagonal())
        else:
            errs=None

        plot_cross = False
        plot_auto = False
        plot_pairs = []

        if plot_corr == 'all':
            # Plot the auto-correlation and the cross-correlation
            plot_cross = True
            plot_auto = True
        elif plot_corr == 'cross':
            # Plot ALL cross-correlations only
            plot_cross = True
        elif plot_corr == 'auto':
            # Plot the auto-correlation only
            plot_auto = True
        elif hasattr(plot_corr, '__iter__'):
            plot_pairs = plot_corr
        else:
            print('plot_corr needs to be \'all\', \'auto\',\'cross\', or a list of pairs of values.')

        tracer_array = np.arange(len(sacc.tracers))
        if plot_cross:
            for tr_i in tracer_array:
                other_tr = np.delete(tracer_array, np.where(tracer_array != tr_i))
                for tr_j in other_tr:
                    # Generate the appropriate list of tracer combinations to plot
                    plot_pairs.append([tr_i, tr_j])

        if plot_auto:
            for tr_i in tracer_array:
                plot_pairs.append([tr_i, tr_i])

        plot_pairs = np.array(plot_pairs)

        ###################################
        # Plotting routines below this line
        ###################################

        npairs = len(tracer_array)
        cmap1 = matplotlib.cm.get_cmap('plasma')
        colors = [cmap1(i) for i in np.linspace(0, 0.9, npairs)]

        # lmaxs = [217.95118625,  258.3063891,   319.73071908,  398.08281899]
        # lmaxs = [ 327.17677938,  387.70958365,  479.84607862,  597.37422849]

        i = 0
        for (tr_i, tr_j) in plot_pairs:
            tbin = np.logical_and(sacc.binning.binar['T1']==tr_i,sacc.binning.binar['T2']==tr_j)
            ell = sacc.binning.binar['ls'][tbin]
            C_ell = sacc.mean.vector[tbin]

            subplot.plot(ell,C_ell * np.power(ell,weightpow),color=colors[i])
            if errs is not None:
                subplot.errorbar(ell,C_ell * np.power(ell,weightpow),yerr=errs[tbin]*np.power(ell,weightpow), linestyle='None',color=colors[i])
            subplot.plot(ell,C_ell * np.power(ell,weightpow),linestyle='None', marker='o', markeredgecolor=colors[i], color=colors[i],
                label= sacc.tracers[0].exp_sample+' $C_{%i%i}$' %(tr_i,tr_j))
            # if errs is not None:
            #     subplot.errorbar(ell[ell<=lmaxs[i]],C_ell[ell<=lmaxs[i]] * np.power(ell,weightpow)[ell<=lmaxs[i]],yerr=errs[tbin][ell<=lmaxs[i]]*np.power(ell,weightpow)[ell<=lmaxs[i]], linestyle='None',color=colors[i])
            #     subplot.errorbar(ell[ell>lmaxs[i]],C_ell[ell>lmaxs[i]] * np.power(ell,weightpow)[ell>lmaxs[i]],yerr=errs[tbin][ell>lmaxs[i]]*np.power(ell,weightpow)[ell>lmaxs[i]], linestyle='None',color=colors[i], alpha=0.3)
            # subplot.plot(ell[ell<=lmaxs[i]],C_ell[ell<=lmaxs[i]] * np.power(ell,weightpow)[ell<=lmaxs[i]],linestyle='None', marker='o', markeredgecolor=colors[i], color=colors[i],
            #     label= sacc.tracers[0].exp_sample+' $C_{%i%i}$' %(tr_i,tr_j))
            # subplot.plot(ell[ell>lmaxs[i]],C_ell[ell>lmaxs[i]] * np.power(ell,weightpow)[ell>lmaxs[i]],linestyle='None', marker='o', markeredgecolor=colors[i], color=colors[i], alpha=0.3)
            if prediction is not None:
                subplot.plot(ell,prediction[tbin] * np.power(ell,weightpow), linestyle=linestyle_pred, color=colors[i])
            i += 1

        if set_logx:
            subplot.set_xscale('log')
        if set_logy:
            subplot.set_yscale('log')
        if show_axislabels:
            subplot.set_xlabel(r'$l$')
            if weightpow == 0:
                elltext = ''
            elif weightpow == 1:
                elltext = r'$\ell$'
            else:
                elltext = r'$\ell^' + '{%i}$' % weightpow
            subplot.set_ylabel(elltext + r'$C_{l}$')
        if show_legend:
            subplot.legend(loc='best')

    def plotData(self, path2fig=None):

        fig = plt.figure()
        subplot=fig.add_subplot(111)

        for i,s in enumerate(self.saccs):
            #plt.subplot(3,3,i+1)
            self.plot_vector(s, subplot,plot_corr = 'auto',weightpow=0,
                          label=self.saccs[0].tracers[0].name, show_axislabels = True, show_legend=True)
            #plt.title(s.tracers[0].name)
        if path2fig is not None:
            plt.savefig(path2fig)
        plt.show()

    def minimize(self):

        res = scipy.optimize.minimize(lambda x:-self.logprobs(x).sum(),
                                self.P.values(),
                                bounds=self.P.bounds(),
                                method='TNC', options={'eps':1e-3, 'disp':True, 'maxiter': 500})

        return res

    def MCMCSample(self,fno='chain'):
        #Setup sampler
        nwalkers=100
        nsteps_burn=100
        nsteps_per_chain=1000
        npar=len(self.P)
        sampler=mc.EnsembleSampler(nwalkers,npar,self.logprob)

        p0=self.P.values()
        #First sample for each walker
        par0=(p0)[None,:]*(1+0.1*np.random.randn(nwalkers,npar))#[:,None]
        #Burning phase
        self.log.info ("Burning")
        pos,prob,stat=sampler.run_mcmc(par0,nsteps_burn)
        sampler.reset()
        #Running
        self.log.info ("Running")
        sampler.run_mcmc(pos,nsteps_per_chain)
        self.log.info("Mean acceptance fraction: {0:.3f}"
              .format(np.mean(sampler.acceptance_fraction)))


        #Save result and analyze it
        np.save(fno,sampler.chain)
        chain=np.load(fno+".npy")
        samples=(chain)[:,nsteps_per_chain/2:,:].reshape((-1,npar))
        for l,m,s in zip(labels,np.mean(samples,axis=0),np.std(samples,axis=0)) :
            print (l+'= %lE +-'%m+' %lE'%s)
        #fig=cn.corner(samples,labels=labels,truths=p0)
        #plt.show()                 




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Calculate HSC clustering cls.')

    parser.add_argument('--path2fig', dest='path2fig', type=str, help='Path to figure.', required=False)
    parser.add_argument('--fitBias', dest='fitBias', type=int, help='Tag denoting if to fit for bias parameters.', required=False, default=1)
    parser.add_argument('--BiasMod', dest='BiasMod', type=str, help='Tag denoting which bias model to use. BiasMod = {bz, const}.', required=False, default='bz')
    parser.add_argument('--fitNoise', dest='fitNoise', type=int, help='Tag denoting if to fit shot noise.', required=False, default=1)
    parser.add_argument('--noiseValue', dest='noiseValue', type=float, help='Constant shot noise value for all bins [in units of 1e-8].', required=False, default=0.75)
    parser.add_argument('--noiseFromData', dest='noiseFromData', type=int, help='Tag denoting if to determine the shot noise from data.', required=False, default=0)
    parser.add_argument('--lmin', dest='lmin', type=str, help='Tag specifying how lmin is determined. lmin = {auto, kmax}.', required=False, default='auto')
    parser.add_argument('--lmax', dest='lmax', type=str, help='Tag specifying how lmax is determined. lmax = {auto, kmax}.', required=False, default='auto')
    parser.add_argument('--kmax', dest='kmax', type=float, help='If lmax=kmax, this sets kmax to use.', required=False)
    parser.add_argument('--fitdata', dest='fitdata', type=int, help='Tag denoting if parameters are fit to data or only a plot is generated.', required=True)
    parser.add_argument('--hod', dest='hod', type=int, help='Tag denoting if to use HOD in theory predictions.', required=False, default=0)
    parser.add_argument('--fitHOD', dest='fitHOD', type=int, help='Tag denoting if to fit for HOD parameters.', required=False, default=0)
    parser.add_argument('--saccfiles', dest='saccfiles', nargs='+', help='Path to saccfiles.', required=True)

    args = parser.parse_args()

    output_filename, _ = os.path.splitext(args.path2fig)

    # Setup logging to file and stderr
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(output_filename+'.txt')
    consoleHandler.setLevel(logging.INFO)
    fileHandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    logger.propagate = False

    if args.lmax == 'kmax':
        zeff = np.array([0.57, 0.70, 0.92, 1.25])
    else:
        zeff = None

    if args.fitBias == 1:
        logger.info('Fitting galaxy bias.')
        if args.BiasMod == 'bz':
            bias = np.array([0.7, 1.5, 1.8, 2.0, 2.5])
        elif args.BiasMod == 'const':
            bias = np.array([0.7, 1.5, 1.8, 2.0])
            # bias = np.array([1., 1., 1., 1.])
        else:
            raise NotImplementedError('Only BiasMod = bz or const implemented.')
    else:
        logger.info('Not fitting galaxy bias.')
        logger.info('Setting bias to 1 for all samples.')
        args.BiasMod = 'const'
        bias = np.ones(4)

    if args.hod == 1:
        # Fiducial starting parameters loosely informed from Coupon et al., 2012
        # arXiv: 1107.0616
        # Table B1
        hodpars = np.array([10., 0., 0.35, 0.3, 10**7.5, 0.7, 10**13., 0.1, 1., 0.3, 0.25, -1.5])
        # hodpars = np.array([10., 0., 0.31, 0., 1e12, 0.3, 3.5e12, 0.2, 0.8, 0.3, 1., -0.1])
    else:
        hodpars=None

    if args.fitdata == 0:
        bias = np.array([0.9615482,  1.16666415, 1.34743442, 1.50652355])
        h = HSCAnalyze(args.saccfiles, Oc=0.258, bias=bias,
                 fitNoise=args.fitNoise, noise=None, fitBias=bool(args.fitBias), BiasMod=args.BiasMod)

        h.plotDataTheory(path2fig=args.path2fig)
        # h.plotDataTheory()

    else:

        if args.fitNoise == 0 and args.noiseFromData == 1:
            h = HSCAnalyze(args.saccfiles, lmax=args.lmax, lmin=args.lmin, kmax=args.kmax, cosmo=None, BiasMod=args.BiasMod,
                           bias=bias, zeff=zeff, fitNoise=args.fitNoise, noise=None, hod=args.hod, fitHOD=args.fitHOD,
                           hodpars=hodpars, fitBias=bool(args.fitBias))
        else:
            h = HSCAnalyze(args.saccfiles, lmax=args.lmax, lmin=args.lmin, kmax=args.kmax, cosmo=None, BiasMod=args.BiasMod,
                           bias=bias, zeff=zeff, fitNoise=args.fitNoise, hod=args.hod, fitHOD=args.fitHOD, hodpars=hodpars,
                           noise=args.noiseValue, fitBias=bool(args.fitBias))
        # h=HSCAnalyze(sys.argv[1:], lmax='kmax', lmin='kmax', kmax=0.15, cosmo=None, \
        #              zeff=np.array([0.57, 0.70, 0.92, 1.25]), fitNoise=False, noise=None)
        # h=HSCAnalyze(sys.argv[1:], BiasMod='const', bias=[0.7,1.5,1.8,2.0], fitNoise=False, noise=None)
        # h=HSCAnalyze(sys.argv[1:])
        res = h.minimize()
        logger.info('Optimizer message {}.'.format(res.message))
        logger.info('Minimum found at {}.'.format(res.x))
        logger.info('Minimum chi2 = {}.'.format(h.chisq_cur))
        logger.info('No of iterations {}.'.format(res.nit))
        h.plotDataTheory(params=res.x, path2fig=args.path2fig)
        # h.plotDataTheory(params=res.x, path2fig='/Users/Andrina/Documents/WORK/HSC-LSS/plots/spectra_eab_best_pzb4bins_bpw200_covdata_cont_dpt_dst_str_ams_fwh_ssk_ssc/cls_data-theory+SN-rem_mPk=halofit_n_sn=const_bz=const_Ntomo=4_kmax=0.15_test-fit.pdf')
        # h.plotDataTheory(params=res.x)

        # h.plotData(path2fig='/Users/Andrina/Documents/WORK/HSC-LSS/plots/spectra_eab_best_pzb4bins_bpw200_covdata_nocont/cls_data_Ntomo=4_lmax=default.pdf')
        #h.MCMCSample()




