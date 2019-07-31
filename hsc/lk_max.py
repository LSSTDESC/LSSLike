import numpy as np
import sacc
import pyccl as ccl

def cutLranges(saccs, lmin, lmax, kmax, zeff, cosmo, Ntomo, logger, saccs_noise=None):
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

        lmax = kmax2lmax(kmax, zeff,logger = logger, cosmo=cosmo)

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

def kmax2lmax(kmax, zeff, logger, cosmo=None):
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