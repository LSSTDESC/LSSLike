import numpy as np
import pyccl as ccl
import sacc
from scipy.interpolate import interp1d

class LSSTheory(object):

    def __init__(self, sacc_in, interp=False,
                 lmax_interp=None, ells_to_interp=None, ells_to_eval=None):
        """

        Required Inputs
        ---------------
        * sacc_in: sacc object to use to set up theory -- ASSUMING that each
                   ls/cls for all tracers are binned the same (i.e. no lmax clipping.).

        Optional Inputs
        ---------------
        * interp: bool: set to True to interpolate the predictions.
                        Default: False
        * lmax_interp: None or int: maximum ell to consider when interpolating.
                                    If None, lmax would be max(ells) in the sacc file.
                                    Default: None
        * ells_to_interp: None or arr: array of ells to interpolate on.
                                       If interp is True and ells_to_interp is False,
                                       ells_to_interp would be evenly spaced on
                                       a log scale (using np.geomspace) between 0 and lmax.
                                       Default: None
        * ells_to_eval: arr: array of ells to evaluate for output when interpolating.
                             If None, np.arange(lmax_interp) will be used.
                             Default: None
                             (Note: nmt needs ell_binning for 1 to windowed -- so if
                             the output would be windowed, leave ells_to_eval to be None)
        """
        if  type(sacc_in) == str:
            self.s = sacc.Sacc.load_fits(sacc_in)
        self.interp = interp

        # set up ells
        # first check to make sure that all cls/cls in the sacc file are binned the same.
        for i, tr1, tr2 in enumerate(self.s.get_tracer_combinations()):
            if i == 0:
                ells_base, _ = self.s.get_ell_cl(sacc.standard_types.galaxy_density_cl, tr1, tr2)
            else:
                ells_here, _ = self.s.get_ell_cl(sacc.standard_types.galaxy_density_cl, tr1, tr2)[0]
                if ells_base != ells_here:
                    raise ValueError(f'expect all tracers to be binned the same: have {ells_base} + '
                                     f'{ells_base}'
                                     )
        self.ells = ells_base
        # set up for interpolation
        if self.interp:
            if lmax_interp is None:
                lmax_interp = max(self.ells)
            # set up sparser ells
            if ells_to_interp is None:
                self.ells_fast = np.unique(np.geomspace(0.1, lmax_interp+1).astype(np.int))
            else:
                self.ells_fast = ells_to_interp
            # set up ells to evaluate
            if ells_to_eval is None:
                self.ells_to_eval = np.arange(lmax_interp)
            else:
                self.ells_to_eval = ells_to_eval
            # nells
            self.nells = len(self.ells_to_eval)
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
                z_b_arr = dic_par['gals_z_b'][tr_index]
                b_b_arr = dic_par['gals_b_b'][tr_index]
            except:
                raise ValueError("bias needed for each tracer")

            # get the zbins array
            zbins = thistracer.z

            # set up the bias array to pass to CCL
            # first check data type for z-array for bias and the bias array
            if not isinstance(z_b_arr, float) and (len(z_b_arr) == len(zbins)):
                if np.any(z_b_arr != zbins):
                    raise ValueError('something isnt right: z_b_arr should match the z_arr for dndz.')
                # i.e. input bias is for all z
                bias = b_b_arr
            else:
                # dont want to interpolate or ignore bias evolution (unless explicitly inputted)
                raise ValueError('please include full bias array as intended.')
                # contruct the bias array
                #bias = b_b_arr * np.ones_like(zbins)   # <-- this will affect results since it ignores bias evolution.
            # construct the tracer object
            tr_out[key] = ccl.NumberCountsTracer(cosmo=cosmo, has_rsd=has_rsd, #has_magnification,
                                                 dndz=(zbins, thistracer.nz), bias=(zbins, bias)
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
        theory_out = {}
        cosmo = self.get_cosmo(dic_par)
        tr = self.get_tracers(cosmo, dic_par)

        for i in range(self.nzbins):
            theory_out[i] = {}
            for j in range(self.nzbins):
                tr1, tr2 = 'bin_%s' % i, 'bin_%s' % j
                if self.interp:
                    # use reduced-ell spacing to get the theory prediction
                    # and then interpolate to get the cls for ells needed
                    c_ells_fast = ccl.angular_cl(cosmo, tr[tr1], tr[tr2], self.ells_fast)
                    cls_spline = interp1d(self.ells_fast, c_ells_fast, kind='cubic')
                    c_ells = cls_spline(self.ells_to_eval)
                else:
                    c_ells = ccl.angular_cl(cosmo=cosmo, cltracer1=tr[tr1], cltracer2=tr[tr2], ell=self.ells)

                # save the cells for return
                theory_out[i][j] = c_ells

        return theory_out
