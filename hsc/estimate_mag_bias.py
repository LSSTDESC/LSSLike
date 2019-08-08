#! /usr/bin/env python

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import pyccl as ccl

# Luminosity function parameters from Gabasch et al., 2006 (astro-ph/0510339)
phi0 = 0.0034
M0 = -21.97
a1 = -0.85
a2 = -0.66
alpha = -1.33

gamma = 2.7

def phi(z, M):
    """
    Compute Schechter luminosity function.
    :param z: redshift
    :param M: absolute magnitude
    :return: number density of galaxies of absolute magnitude M at redshift z
    """

    phiz = phi0*(1 + z)**a2
    Mz = M0 + a1*np.log(1 + z)

    Mtemp = 10**(0.4*(Mz - M))
    phi = 0.4*np.log(10)*phiz*Mtemp**(alpha + 1)*np.exp(-Mtemp)

    return phi

def ng(z, Mlim):
    """
    Cumulative number density of galaxies with M < Mlim at redshift z.
    :param z: redshift
    :param Mlim: limiting absolute magnitude
    :return: cumulative number density of galaxies with M < Mlim at redshift z
    """

    Mgrid = np.linspace(-10000., Mlim, 10000)
    phigrid = phi(z, Mgrid)

    ng = np.trapz(phigrid, Mgrid)

    return ng

def K(z):
    """
    K-correction for i' band cenetered at 770 nm, assuming a spectrum F_nu propto nu^-gamma.
    This follows the implementation in Montanari & Durrer, 2015, arXiv:1506.01369
    :param z:
    :return:
    """

    K = 2.5*(gamma - 1.)*np.log10(1 + z)

    return K

def s(cosmo, z, mlim):
    """
    Magnification bias for limiting apparent magnitude mlim at redshift z.
    :param cosmo: CCL cosmology object
    :param z: redshift
    :param mlim: limiting apparent magnitude
    :return: magnification bias for limiting apparent magnitude mlim at redshift z
    """

    a = 1./(1 + z)
    # Convert from apparent to absolute magnitude
    Mlim = mlim - 5.*np.log10(ccl.luminosity_distance(cosmo, a)*10.**5) - K(z)

    s = 1./np.log(10)*phi(z, Mlim)/ng(z, Mlim)

    return s

def angular_cl(cosmo, tracer1, tracer2, ells, pk_mm, pk_gg, pk_gm):
    """
    Compute angular cl including magnification bias with the generalized tracer framework of CCL.
    :param cosmo: CCL cosmology object
    :param tracer1: list of [g_tracer, m_tracer] for tracer 1
    :param tracer2: list of [g_tracer, m_tracer] for tracer 2
    :param ells: angular multipole range to compute cls
    :param pk_mm: 3D matter power spectrum computed using halo model
    :param pk_gg: 3D galaxy power spectrum computed using HOD
    :param pk_gm: 3D galaxy-matter cross power spectrum computed using halo model and HOD
    :return cl: spherical harmonic power spectrum with magnification bias for ells
    """

    g_tracer1, m_tracer1 = tracer1
    g_tracer2, m_tracer2 = tracer2

    cl_gg = ccl.angular_cl(cosmo, g_tracer1, g_tracer2, ells, p_of_k_a=pk_gg)
    cl_mm = ccl.angular_cl(cosmo, m_tracer1, m_tracer2, ells, p_of_k_a=pk_mm)
    cl_gm = ccl.angular_cl(cosmo, g_tracer1, m_tracer2, ells, p_of_k_a=pk_gm)
    cl_mg = ccl.angular_cl(cosmo, m_tracer1, g_tracer2, ells, p_of_k_a=pk_gm)

    cl = cl_gg + cl_gm + cl_mg + cl_mm

    return cl

