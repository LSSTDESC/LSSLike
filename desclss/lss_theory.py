import numpy as np
import pyccl as ccl
import sacc
from scipy.interpolate import interp1d

class LSSTheory(object):

    def __init__(self,sacc_pth):
        if  type(sacc_pth)==str:
            self.s=sacc.Sacc.load_fits(sacc_pth)
        self.s=sacc_pth
        #if self.s.binning==None :
        #    raise ValueError("Binning needed!")

    def get_tracers(self,cosmo,dic_par) :
        tr_out=[]
        has_rsd=dic_par.get('has_rsd',False)
        has_magnification=dic_par.get('has_magnification',False)
        for (tr_index, thistracer) in enumerate(self.s.tracers.values()) :
            #if thistracer.type.__contains__('point'):
            #try:
            #print(thistracer.tracer_type+'_b_b', dic_par.keys())
            #z_b_arr=dic_par[thistracer.tracer_type+'_z_b'][tr_index]
            b_b_arr=dic_par[thistracer.tracer_type+'_b_b'][tr_index]
            #except:
            #    raise ValueError("bias needed for each tracer")
            if 'zshift_bin' + str(tr_index) in dic_par:
                zbins = thistracer.z + dic_par['zshift_bin' + str(tr_index)]
            else:
                zbins = thistracer.z 
        
            #bf=interp1d(z_b_arr,b_b_arr,kind='nearest') #Assuming linear interpolation. Decide on extrapolation.
            b_arr=b_b_arr*np.ones_like(thistracer.z) #Constant bias in the bin
            tr_out.append(ccl.NumberCountsTracer(cosmo, dic_par['has_rsd'],
                                                       (zbins, thistracer.nz), (zbins, b_arr)))
            #else :
            #    raise ValueError("Only \"point\" tracers supported")
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
        has_sigma8 = ('sigma8' in dic_par)
        has_A_s = ('A_s' in dic_par)
        transfer_function=dic_par.get('transfer_function','boltzmann_class')
        matter_power_spectrum=dic_par.get('matter_power_spectrum','halofit')
        if has_sigma8 and has_A_s:
            raise ValueError("Specifying both sigma8 and A_s: pick one")
        elif has_A_s:
            A_s = dic_par['A_s']
            cosmo=ccl.Cosmology(Omega_c=Omega_c,Omega_b=Omega_b,Omega_k=Omega_k,
                                  w0=w,wa=wa,A_s=A_s,n_s=n_s,h=h0,
                                  transfer_function=transfer_function,
                                  matter_power_spectrum=matter_power_spectrum)

        else:
            sigma8=dic_par.get('sigma8',0.8)
            cosmo=ccl.Cosmology(Omega_c=Omega_c,Omega_b=Omega_b,Omega_k=Omega_k,
                                  w0=w,wa=wa,sigma8=sigma8,n_s=n_s,h=h0)

        return cosmo

    def get_prediction(self,dic_par) :
        theory_out=[]
        cosmo=self.get_cosmo(dic_par)
        tr=self.get_tracers(cosmo, dic_par)
        for b1, b2 in self.s.get_tracer_combinations():
            ells, _, covmat = self.s.get_ell_cl('galaxy_density_cl', b1, b2, return_cov=True)
            i1 = int(b1.split('_')[-1])
            i2 = int(b2.split('_')[-1])
            cls=ccl.angular_cl(cosmo, tr[i1], tr[i2], ells)
            if (i1==i2) and ('Pw_bin%i'%i1 in dic_par):
                cls+=dic_par['Pw_bin%i'%i1]
            theory_out.append(cls)
            
        return np.array(np.concatenate(theory_out).ravel())    
