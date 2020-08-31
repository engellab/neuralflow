cimport numpy as np

def getGamma0(np.ndarray[np.float64_t, ndim=2] G,
    int Nv, np.ndarray[np.float64_t, ndim=1] lQd,
    np.ndarray[np.float64_t, ndim=1] tempExp,
    np.ndarray[np.float64_t, ndim=2] alpha,
    np.ndarray[np.float64_t, ndim=1] btemp,
    double dt, int i):
    """Gamma-function needed for gradient calculation, see Eq. (36) in Supplementary Information
    This is the most computationally intensive step implemented in Cython, which takes advantage of C performance  
    """
    
    
    cdef int j, k

    for j in range(Nv):
        for k in range(Nv):
            if j==k:
                G[j,k] += alpha[j, i]*btemp[j]*dt*tempExp[j]
            else:
                G[j,k] += alpha[j, i]*btemp[k]*(tempExp[j]-tempExp[k])/(lQd[k]-lQd[j])

