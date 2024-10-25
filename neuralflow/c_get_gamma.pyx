"""This source file contains cython implementation of G and G0 functions.
"""
cimport numpy as np

def G1_d(np.ndarray[np.float64_t, ndim=3] GI,
    int Nv, np.ndarray[np.float64_t, ndim=1] tempExp,
    np.ndarray[np.float64_t, ndim=2] alpha,
    np.ndarray[np.float64_t, ndim=1] btemp,
    int i, int nnum):
    """
    Gamma function needed for gradients w.r.t. FR and C (for double
    precision calculations)
    """

    cdef int j, k
    for j in range(Nv):
        for k in range(Nv):
            GI[j, k, nnum]+=alpha[j,i] * tempExp[j]*btemp[k]

def G0_d(np.ndarray[np.float64_t, ndim=2] G,
    int Nv, np.ndarray[np.float64_t, ndim=1] lQd,
    np.ndarray[np.float64_t, ndim=1] tempExp,
    np.ndarray[np.float64_t, ndim=2] alpha,
    np.ndarray[np.float64_t, ndim=1] btemp,
    double dt, int i):
    """
    Gamma-function needed for all of the gradients (for double
    precision calculations) 
    """
        
    cdef int j, k
    for j in range(Nv):
        for k in range(Nv):
            if j==k:
                G[j,k] += alpha[j, i] * btemp[j] * dt * tempExp[j]
            else:
                G[j,k] += alpha[j, i] * btemp[k] * (tempExp[j] - tempExp[k]) / (lQd[k] - lQd[j])