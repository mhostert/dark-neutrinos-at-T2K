#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#cython: boundscheck=False
#cython: language_level=3
#cython: wraparound=False
#cython: nonecheck=False
#cython: define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#cython: cdivision=True

cimport vegas
import vegas

import numpy as np
cimport numpy as np
from numpy cimport ndarray
#CYTHON
import pyximport
pyximport.install(
    language_level=3,
    pyimport=False,
    )
from . import Cfourvec as Cfv

from collections import defaultdict
from functools import partial

#######################################
# C functions to be used
from libc.math cimport sqrt, abs, log, cos, sin, acos, exp, M_PI

#######################################
# get samples from VEGAS integration and their respective weights
def get_samples(int DIM, object integ, object batch_f):
    
        # cdef ndarray[double, ndim=1] SAMPLES = np.empty((1))
        # cdef ndarray[double, ndim=1] weights = np.empty((1))
        
        SAMPLES = [[] for i in range(DIM)]
        weights = defaultdict(partial(np.ndarray,0))
        for x, wgt in integ.random_batch():
            for regime in batch_f(x).keys():
                weights[regime] = np.append(weights[regime], wgt*(batch_f(x)[regime]))
            # weights['pel'] = np.append(weights['pel'], wgt*(batch_f(x)['pel']))
            # weights['coh'] = np.append(weights['coh'], wgt*(batch_f(x)['coh']))
            for i in range(DIM):
                SAMPLES[i] = np.concatenate((SAMPLES[i],x[:,i]))

        return np.array(SAMPLES), weights
