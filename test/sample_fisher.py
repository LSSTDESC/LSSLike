#!/usr/bin/env python
import sys
import numpy as np
import sacc
sys.path.append('/users/boryanah/repos/LSSLike/desclss/')
from desclss import LSSTheory
import copy

#Simulation to use
if len(sys.argv)!=2 :
    print("Usage ./sample_emcee.py sim_file")
    exit(1)
sim_file=sys.argv[1]

#Read file with noiseless sim and initialize theory object
s0=sacc.SACC.loadFromHDF(sim_file)
lt=LSSTheory(sim_file)

#Parameters
params={}
params['omh2']={'value':0.1414,'dval':0.005,'label':'$\\Omega_M\\,h^2$','isfree':True}
params['w']   ={'value':-1.0  ,'dval':0.02 ,'label':'$w$'              ,'isfree':True}
params['s8']  ={'value':0.8   ,'dval':0.02 ,'label':'$\\sigma_8$'      ,'isfree':False}
for i in np.arange(4) :
    params['b%d'%i]  ={'value':1.0   ,'dval':0.02 ,'label':'$b_%d$'%i  ,'isfree':True}
npar=len(params)
params_vary={}
for p in params :
    if params[p]['isfree'] :
        params_vary[p]=params[p]
npar_vary=len(params_vary)

#Theory from parameters
def compute_theory(par) :
    oc=par['omh2']['value']/0.67**2-0.045
    dic={'omega_c':oc,'omega_b':0.045,'omega_k':0.0,'omega_nu':0.0,'w':par['w']['value'],
         'h0':0.67,'sigma_8':par['s8']['value'],'n_s':0.96,
         'transfer_function':'eisenstein_hu','matter_power_spectrum':'linear'}
    dic.update({'gals_z_b':[0.0,0.5,0.7,0.9,1.1,1.7],
                'gals_b_b':[1.0,par['b0']['value'],par['b1']['value'],
                            par['b2']['value'],par['b3']['value'],1.0]})
    return lt.get_prediction(dic)

#Compute derivatives of the data vector
dcl=np.zeros([npar_vary,len(s0.mean.vector)])
def compute_derivative(pars,parname) :
    pars_here=copy.deepcopy(pars)
    pars_here[parname]['value']=pars[parname]['value']+pars[parname]['dval']
    clp=compute_theory(pars_here)
    pars_here[parname]['value']=pars[parname]['value']-pars[parname]['dval']
    clm=compute_theory(pars_here)

    return (clp-clm)/(2*pars[parname]['dval'])
                      
for i,nam in enumerate(params_vary) :
    dcl[i,:]=compute_derivative(params,nam)

#Compute Fisher matrix, covariance and correlation matrix
fisher=np.dot(dcl,np.dot(s0.precision.matrix,dcl.T))
np.save('test/chains_om_w_bnn/fisher',fisher)
