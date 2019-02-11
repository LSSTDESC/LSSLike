import numpy as np
import pyccl as ccl
from desclss.hod import HODProfile
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm

def lmminf(z) :
    #Returns log10(M_min)
    return 11.+0.5*(1+z)

sigmf=lambda x: 0.31

def m0f(z) :
    #Returns M_0
    return 1E12*(1+z)**0.3

def m1f(z) :
    #Returns M_1
    return 3.5E12*(1+z)**0.2

def alphaf(z) :
    #Returns alpha
    return 0.8*(1+z)**0.3

def fcf(z) :
    #Returns f_central
    return 1./(1+z)**0.1

##############################################################
#                                                            #
# This script illustrates how to generate predictions        #
# for the angular power spectrum of HOD models.              #
# Note that this only works with an unofficial branch        #
# of CCL: https://github.com/LSSTDESC/CCL/tree/lss_hsc_work  #
#                                                            #
##############################################################


#Initialize CCL cosmology
cosmo=ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96)
karr=np.logspace(-4.,2.,512)
zarr=np.linspace(0.,3.,64)[::-1]

#Initialize HOD profile
hod=HODProfile(cosmo,lmminf,sigmf,fcf,m0f,m1f,alphaf)


#Compute power spectrum at a given redshift (just for illustrative purposes)
pkarr,p1harr,p2harr,nkarr,bkarr=hod.pk(karr,0.5,return_decomposed=True)
#Plot for fun
plt.figure()
plt.plot(karr,p1harr,'r-',label='1-halo')
plt.plot(karr,p2harr,'b-',label='2-halo')
plt.plot(karr,pkarr,'k-',label='Total')
plt.plot(karr,nkarr,'k--',lw=1,label='Shot noise')
plt.legend(loc='lower left')
plt.xlim([1E-4,1E2])
plt.loglog()
plt.xlabel('$k\\,\\,[{\\rm Mpc}^{-1}]$',fontsize=15)
plt.ylabel('$P(k)\\,\\,[{\\rm Mpc}^{-1}]$',fontsize=15)

#Compute array of power spectra at different redshifts
pk_z_arr=np.log(np.array([hod.pk(karr,1/(1+z)) for z in zarr]))
#Plot for fun
plt.figure()
for ip,p in enumerate(pk_z_arr) :
    plt.plot(karr,np.exp(p),c=cm.winter(ip/(len(zarr)-1.)))
plt.xlim([1E-4,1E2])
plt.loglog()
plt.xlabel('$k\\,\\,[{\\rm Mpc}^{-1}]$',fontsize=15)
plt.ylabel('$P(k)\\,\\,[{\\rm Mpc}^{-1}]$',fontsize=15)

#Initialize CCL 2D power spectrum object
pk_hod=ccl.Pk2D(a_arr=1./(1+zarr),lk_arr=np.log(karr),pk_arr=pk_z_arr,is_logp=True)

#Initialize a number counts tracer
z=np.linspace(0.,3.,1024)
nz=np.exp(-0.5*((z-0.5)/0.1)**2)
#Set bias to 1, since biasing is now taken care of by the HOD
bz=np.ones_like(z)
t=ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z,nz),bias=(z,bz))

#Compute angular power spectrum
ell=np.arange(1E4)
#We pass the 2D power spectrum object as p_of_k_a
cell=ccl.angular_cl(cosmo,t,t,ell,p_of_k_a=pk_hod)
#Let's plot the results
plt.figure()
plt.plot(ell,cell)
plt.loglog()
plt.xlabel('$\\ell$',fontsize=16)
plt.ylabel('$C_\\ell$',fontsize=16)
plt.loglog()
plt.show()

