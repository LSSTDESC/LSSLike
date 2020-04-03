import sacc
import numpy as np

#Not implemented:
# - Include ell-cuts?
# - Include cuts in the pairs of tracers to correlate?
# - Ability to return simulated scatter

class LSSLikelihood(object):
    def __init__(self,saccin) :
        if (type(saccin)==type("filename")):
            self.s=sacc.SACC.load_fits(saccin)
        else:
            self.s=saccin
        #if self.s.precision==None :
        #    raise ValueError("Precision matrix needed!")
        #if self.s.mean==None :
        #    raise ValueError("Mean vector needed!")
        
        self.pmatrix = self.s.covariance.inverted()

    #We're assuming data_theory will come in the form of a sacc.Means object
    def __call__(self,theory_vec) :
        return -0.5*self.chi2(theory_vec)

    def chi2(self,theory_vec):
        delta=theory_vec - self.s.mean
        chi2=np.einsum('i,ij,j',delta,self.pmatrix,delta)
#        chi2=np.linalg.multi_dot([delta,self.s.precision.matrix,delta])
        return chi2
