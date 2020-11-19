import sacc
import numpy as np

# Not implemented:
# - Include ell-cuts?
# - Include cuts in the pairs of tracers to correlate?
# - Ability to return simulated scatter

class LSSLikelihood(object):
    def __init__(self, saccin) :
        if (type(saccin) == type("filename")):
            self.s = sacc.Sacc.load_fits(saccin)
        else:
            self.s = saccin
        if self.s.covariance is None :
            raise ValueError("Covariance matrix needed!")
        if self.s.mean is None:
            raise ValueError("Mean vector needed!")

    def __call__(self, theory_vec) :
        return -0.5 * self.chi2(theory_vec)

    def chi2(self, theory_vec):
        data_vec = self.s.mean
        cov_type = self.s.covariance.cov_type
        if cov_type == 'diagonal':
            covmat = np.diag(self.s.covariance.diag)
        elif cov_type == 'full':
            covmat = self.s.covariance.covmat
        else:
            raise ValueError('something is wrong. cov type not diagonal or full but %s' % cov_type)

        delta = theory_vec - data_vec
        # assume that the covariance is invertible
        pmatrix = np.linalg.pinv(covmat)
        # calcualte chi2
        chi2 = np.linalg.multi_dot([delta, pmatrix, delta])
        return chi2
