import numpy as np
import pyccl as ccl
import logging
from scipy.interpolate import interp1d
import hod
import hod_funcs

HOD_PARAM_KEYS = ['lmmin_0', 'lmmin_alpha', 'sigm_0', 'sigm_alpha', 'm0_0', 'm0_alpha', 'm1_0', 'm1_alpha', \
                  'alpha_0', 'alpha_alpha', 'fc_0', 'fc_alpha']

class LSSTheory(object):

    def __init__(self,sacc, log=logging.INFO, hod=False, fitHOD=False, hodpars=None):
        """

        :param sacc:
        :param log:
        :param hod: if hod=True, use HOD for theory predictions, otherwise use normal built-in CCL functions
        :return:
        """
        if  type(sacc)==str:
            sacc=sacc.SACC.loadFromHDF(sacc)
        self.s=sacc
        if self.s.binning==None :
            raise ValueError("Binning needed!")

        if type(log)==logging.Logger:
            self.log=log
        else:
            self.log = logging.getLogger('LSSTheory')
            self.log.setLevel(log)
            ch = logging.StreamHandler()
            ch.setLevel(log)
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            ch.setFormatter(formatter)
            self.log.addHandler(ch)
            print (self.log)

        # Set HOD flag
        self.hod = hod
        self.fitHOD = fitHOD

        if hod == 1:
            self.log.info('Using HOD for theoretical predictions.')
            assert hodpars is not None, 'Using HOD for theoretical predictions but no HOD parameter values supplied. Aborting.'
            self.z_arr = np.linspace(0., 3., 500)[::-1]
            self.a_arr = 1./(1. + self.z_arr)
            # self.a_arr = np.logspace(0, -1.8, 500)[::-1]
            # self.z_arr = 1./self.a_arr - 1.
            self.k_arr = np.logspace(-4, 2, 1000)
            dic_hodpars = dict(zip(HOD_PARAM_KEYS, hodpars))
            self.hodpars = hod_funcs.HODParams(dic_hodpars)
        else:
            self.log.info('Not using HOD for theoretical predictions.')

    def get_tracers(self,cosmo,dic_par) :
        tr_out=[]
        has_rsd=dic_par.get('has_rds',False)
        has_magnification=dic_par.get('has_magnification',None)
        for (tr_index, thistracer) in enumerate(self.s.tracers) :
            if thistracer.type.__contains__('point'):
                if thistracer.exp_sample+'_b_bin' + str(tr_index) in dic_par:
                    b_b = dic_par[thistracer.exp_sample+'_b_bin' + str(tr_index)]
                    z_b_arr = thistracer.z
                    b_b_arr = b_b*np.ones_like(z_b_arr)
                elif thistracer.exp_sample+'_z_b' in dic_par:
                    z_b_arr=dic_par[thistracer.exp_sample+'_z_b']
                    b_b_arr = dic_par[thistracer.exp_sample+'_b_b']
                    bf=interp1d(z_b_arr,b_b_arr,kind='nearest') #Assuming linear interpolation. Decide on extrapolation.
                    b_arr=bf(thistracer.z) #Assuming that tracers have this attribute
                else:
                    raise ValueError("bias needed for each tracer")

                if 'zshift_bin' + str(tr_index) in dic_par:
                    zbins = thistracer.z + dic_par['zshift_bin' + str(tr_index)]
                else:
                    zbins = thistracer.z

                tr_out.append(ccl.NumberCountsTracer(cosmo, has_rsd=dic_par['has_rsd'], dndz=(zbins, thistracer.Nz), \
                                                     bias=(z_b_arr, b_b_arr), mag_bias=dic_par['has_magnification']))
            else :
                raise ValueError("Only \"point\" tracers supported")
        return tr_out

    def get_cosmo(self,dic_par) :
        Omega_c = dic_par.get('Omega_c',0.255)
        Omega_b = dic_par.get('Omega_b', 0.045)
        Omega_k = dic_par.get('Omega_k', 0.0)
        mnu = dic_par.get('mnu', 0.06)
        w  = dic_par.get('w', -1.0)
        wa = dic_par.get('wa', 0.0)
        h0 = dic_par.get('h0', 0.67)
        n_s = dic_par.get('n_s', 0.96)
        has_sigma8 = ('sigma_8' in dic_par)
        has_A_s = ('A_s' in dic_par)

        transfer_function=dic_par.get('transfer_function','boltzmann_class')
        matter_power_spectrum=dic_par.get('matter_power_spectrum','halofit')

        if has_sigma8 and has_A_s:
            raise ValueError("Specifying both sigma8 and A_s: pick one")
        elif has_A_s:
            A_s = dic_par['A_s']
            cosmo=ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, Omega_k=Omega_k, w0=w, wa=wa, A_s=A_s, n_s=n_s, h=h0,
                                transfer_function=dic_par['transfer_function'],
                                matter_power_spectrum=dic_par['matter_power_spectrum'])

        else:
            sigma8=dic_par.get('sigma_8',0.8)
            cosmo=ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, Omega_k=Omega_k, w0=w, wa=wa, sigma8=sigma8, n_s=n_s, h=h0,
                                transfer_function=dic_par['transfer_function'],
                                matter_power_spectrum=dic_par['matter_power_spectrum'])

        self.log.info('CCL called with cosmology = {}.'.format(cosmo))

        return cosmo

    def get_prediction(self,dic_par) :
        theory_out=np.zeros((self.s.size(),))
        cosmo=self.get_cosmo(dic_par)
        tr=self.get_tracers(cosmo,dic_par)

        if self.fitHOD == 1:
            dic_hodpars = {}
            for key in HOD_PARAM_KEYS:
                dic_hodpars[key] = dic_par[key]
            self.hodpars = hod_funcs.HODParams(dic_hodpars)

        if self.hod == 1:
            hodprof = hod.HODProfile(cosmo, self.hodpars.lmminf, self.hodpars.sigmf, self.hodpars.fcf, self.hodpars.m0f, \
                                         self.hodpars.m1f, self.hodpars.alphaf)
            # pk_hod_arr = np.log(np.array([hodprof.pk(cosmo, z, self.k_arr) for z in self.z_arr]))
            # pk_hod = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=pk_hod_arr, is_logp=True)
            pk_hod = ccl.Pk2D(pkfunc=hodprof.pk, is_logp=False)

        for i1,i2,_,ells,ndx in self.s.sortTracers() :
            if self.hod == 0:
                self.log.info('hod = {}. Not using HOD to compute theory predictions.'.format(self.hod))
                cls=ccl.angular_cl(cosmo,tr[i1],tr[i2],ells)
            else:
                self.log.info('hod = {}. Using HOD to compute theory predictions.'.format(self.hod))
                cls = ccl.angular_cl(cosmo,tr[i1],tr[i2],ells, p_of_k_a=pk_hod)
            if (i1==i2) and ('Pw_bin%i'%i1 in dic_par):
                cls+=dic_par['Pw_bin%i'%i1]
            theory_out[ndx]=cls
            
        return theory_out

    def get_noSN_prediction(self,dic_par) :
        theory_out=np.zeros((self.s.size(),))
        cosmo=self.get_cosmo(dic_par)
        tr=self.get_tracers(cosmo,dic_par)

        if self.fitHOD == 1:
            dic_hodpars = {}
            for key in HOD_PARAM_KEYS:
                dic_hodpars[key] = dic_par[key]
            self.hodpars = hod_funcs.HODParams(dic_hodpars)

        if self.hod == 1:
            hodprof = hod.HODProfile(self.hodpars.lmminf, self.hodpars.sigmf, self.hodpars.fcf, self.hodpars.m0f, \
                                         self.hodpars.m1f, self.hodpars.alphaf)
            pk_hod_arr = np.log(np.array([hodprof.pk(cosmo, z, self.k_arr) for z in self.z_arr]))
            pk_hod = ccl.Pk2D(a_arr=self.a_arr, lk_arr=np.log(self.k_arr), pk_arr=pk_hod_arr, is_logp=True)

        for i1,i2,_,ells,ndx in self.s.sortTracers() :
            if self.hod == 0:
                self.log.info('hod = {}. Not using HOD to compute theory predictions.'.format(self.hod))
                cls=ccl.angular_cl(cosmo,tr[i1],tr[i2],ells)
            else:
                self.log.info('hod = {}. Using HOD to compute theory predictions.'.format(self.hod))
                cls = ccl.angular_cl(cosmo,tr[i1],tr[i2],ells, p_of_k_a=pk_hod)
            theory_out[ndx]=cls

        return theory_out
