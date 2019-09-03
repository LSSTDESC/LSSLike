# LSSLike
LSST DESC single-probe LSS likelihood module

# Purpose
Provide software support to:

1. Store and read generic arrays of 2-point functions and their precision matrices.
2. Compute theoretical predictions for this data vector for a set of cosmological and nuisance parameters in the case of galaxy-galaxy clustering.
3. Evaluate the likelihood of a given set of 2-point measurements.

# Note on SACC

This code requires an old version of sacc library, namely 0.1 [release](https://github.com/LSSTDESC/sacc/releases).
