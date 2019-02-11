#!/usr/bin/env python

import numpy as np
import pyccl as ccl
from desclss.hod import HODProfile
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

#Initialize CCL cosmology
cosmo=ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96)
karr=np.logspace(-4.,2.,512)
zarr=np.linspace(0.,3.,64)[::-1]

#Initialize a number counts tracer
z=np.linspace(0.,3.,1024)
nz=np.exp(-0.5*((z-0.5)/0.1)**2)
#Set bias to 1, since biasing is now taken care of by the HOD
bz=np.ones_like(z)
t=ccl.NumberCountsTracer(cosmo,has_rsd=False,dndz=(z,nz),bias=(z,bz))

plt.figure(figsize=(5,5))

for i in range(7):
    lmmin=12.4
    m0 = 10
    m1 = 14
    sigma= 0.4
    alpha=1.0
    fc=0.9
    if i==1:
        name='lmmin'
        delta=lmmin*0.1
        lmmin+=delta
    elif i==2:
        name='m0'
        delta=m0*0.1
        m0+=delta
    elif i==3:
        name='m1'
        delta=m1*0.1
        m1+=delta
    elif i==4:
        name='sigma'
        delta=sigma*0.1
        sigma+=delta
    elif i==5:
        name='alpha'
        delta=alpha*0.1
        alpha +=delta
    elif i==6:
        name='fc '
        delta=fc *0.1
        fc +=delta

    print ("i=",i,lmmin,sigma,fc,m0,m1,alpha)
        
    #Initialize HOD profile
    hod=HODProfile(cosmo,lambda x:lmmin,lambda x:sigma,lambda x:fc,lambda x:10**m0,lambda x:10**m1,lambda x:alpha)
    pk_z_arr=np.log(np.array([hod.pk(karr,1/(1+z)) for z in zarr]))
    #Initialize CCL 2D power spectrum object
    pk_hod=ccl.Pk2D(a_arr=1./(1+zarr),lk_arr=np.log(karr),pk_arr=pk_z_arr,is_logp=True)
    #Compute angular power spectrum
    ell=np.arange(200,3000)
    #We pass the 2D power spectrum object as p_of_k_a
    cell=ccl.angular_cl(cosmo,t,t,ell,p_of_k_a=pk_hod)
    if i==0:
        cell0=np.copy(cell)
    else:
    #Let's plot the results
        diff=(cell-cell0)/delta/cell0
        fact=1/diff[800]
        plt.plot(ell,diff*fact,label='X='+name+' r=%3.2f'%fact)



plt.semilogx()
plt.xlabel('$\\ell$',fontsize=16)
plt.ylabel('$d log C_\ell/d log X \\times r$',fontsize=16)
plt.legend()
plt.show()

