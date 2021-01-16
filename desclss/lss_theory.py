import numpy as np
import pyccl as ccl
import sacc
from scipy.interpolate import interp1d

class LSSTheory(object):

    def __init__(self, sacc_in, interp=False, lmax=None):
        if  type(sacc_in) == str:
            self.s = sacc.Sacc.load_fits(sacc_in)
        self.interp = interp
        # set up ells
        tr1, tr2 = self.s.get_tracer_combinations()[0]
        self.ells, _ = self.s.get_ell_cl(sacc.standard_types.galaxy_density_cl, tr1, tr2)
        # set up for interpolation
        if self.interp:
            if lmax is None:
                lmax = max(self.ells)
            # set up sparser ells
            self.ells_fast = np.unique(np.geomspace(0.1, lmax+1).astype(np.int))
            # create the new ells to eval
            self.ells_to_eval = np.arange(lmax)
            # nells
            self.nells = lmax
        else:
            self.nells = len(self.ells)
        # number of zbins
        self.nzbins = len(self.s.tracers.keys())

    def get_tracers(self, cosmo, dic_par) :
        tr_out = {}
        has_rsd = dic_par.get('has_rsd', False)
        has_magnification = dic_par.get('has_magnification', False)

        for (tr_index, key) in enumerate(self.s.tracers) :
            thistracer = self.s.tracers[key]
            try:
                b_b_arr = dic_par['gals_b_b'][tr_index]
            except:
                raise ValueError("bias needed for each tracer")

            if 'zshift_bin' + str(tr_index) in dic_par:
                zbins = thistracer.z + dic_par['zshift_bin' + str(tr_index)]
            else:
                zbins = thistracer.z

            tr_out[key] = ccl.NumberCountsTracer(cosmo=cosmo, has_rsd=has_rsd, #has_magnification,
                                                 dndz=(zbins, thistracer.nz), bias=(zbins, b_b_arr * np.ones_like(zbins))
                                                )
        return tr_out

    def get_cosmo(self, dic_par):
        # get the parameter values from the input dictionary
        # if the key isn't assume, assume the value specified here
        Omega_c = dic_par.get('Omega_c', 0.255)
        Omega_b = dic_par.get('Omega_b', 0.045)
        Omega_k = dic_par.get('Omega_k', 0.0)
        mnu = dic_par.get('mnu', 0.06)
        w  = dic_par.get('w', -1.0)
        wa = dic_par.get('wa', 0.0)
        h0 = dic_par.get('h0', 0.67)
        n_s = dic_par.get('n_s', 0.96)
        has_sigma8 = ('sigma_8' in dic_par)
        has_A_s = ('A_s' in dic_par)
        if has_sigma8 and has_A_s:
            raise ValueError("Specifying both sigma8 and A_s: pick one")
        elif has_A_s:
            A_s = dic_par['A_s']
            sigma8 = None
        else:
            A_s = None
            sigma8 = dic_par.get('sigma_8', 0.8)

        transfer_function = dic_par.get('transfer_function', 'boltzmann_class')
        matter_power_spectrum = dic_par.get('matter_power_spectrum', 'halofit')
        # set up the ccl object
        cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, Omega_k=Omega_k,
                              w0=w, wa=wa, A_s=A_s, n_s=n_s, h=h0, sigma8=sigma8,
                              transfer_function=transfer_function,
                              matter_power_spectrum=matter_power_spectrum)
        return cosmo

    def get_prediction(self, dic_par):
        theory_out = np.zeros((self.nzbins, self.nzbins, self.nells))
        cosmo = self.get_cosmo(dic_par)
        tr = self.get_tracers(cosmo, dic_par)

        for i in range(self.nzbins):
            for j in range(self.nzbins):
                tr1, tr2 = 'bin_%s' % i, 'bin_%s' % j
                #ells, _ = self.s.get_ell_cl(sacc.standard_types.galaxy_density_cl, tr1, tr2)
                if self.interp:
                    # use reduced-ell spacing to get the theory prediction
                    # and then interpolate to get the cls for ells needed
                    #ells_fast = np.unique(np.geomspace(0.1, max(ells)+1).astype(np.int))
                    c_ells_fast = ccl.angular_cl(cosmo, tr[tr1], tr[tr2], self.ells_fast)
                    cls_spline = interp1d(self.ells_fast, c_ells_fast, kind='cubic')
                    c_ells = cls_spline(self.ells_to_eval)
                else:
                    c_ells = ccl.angular_cl(cosmo=cosmo, cltracer1=tr[tr1], cltracer2=tr[tr2], ell=self.ells)

                # save the cells for return
                theory_out[i][j] = c_ells

        return theory_out