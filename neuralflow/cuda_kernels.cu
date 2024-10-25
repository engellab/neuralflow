// This file contains Cuda kernels for GPU calculation of the Gamma functions 
// Also see c_get_gamma.pyx for CPU analogues

extern "C" {
 
 __global__ void G0_d_gpu(double* G, const int Nv, const double* alpha,
                     const double* dark_exp, const double* lQd,
                     const double* btemp, const double* dt_arr, const int i){
    // Gamma-function needed for gradients w.r.t. driving force F (for double precision calculations)
    const int numThreadsx = blockDim.x * gridDim.x;
    const int threadIDx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreadsy = blockDim.y * gridDim.y;
    const int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ic = threadIDx*Nv + threadIDy;
    const double dt = dt_arr[0];
    if (threadIDx < Nv && threadIDy < Nv)
        if (threadIDx==threadIDy)
            G[ic]+=alpha[threadIDx+i*Nv]*btemp[threadIDy]*dt*dark_exp[threadIDx+i*Nv];
        else
            G[ic]+=alpha[threadIDx+i*Nv]*btemp[threadIDy]*(dark_exp[threadIDx+i*Nv]-dark_exp[threadIDy+i*Nv])/(lQd[threadIDy]-lQd[threadIDx]);  
    }
    
 __global__ void G1_d_gpu(double* GI, const int Nv, const double* alpha,
                     const double* dark_exp,
            const double* btemp, const int i, const int* nid) {
    // Gamma function needed for gradients w.r.t. firing rates (for double precision calculations)
    const int numThreadsx = blockDim.x * gridDim.x;
    const int threadIDx = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreadsy = blockDim.y * gridDim.y;
    const int threadIDy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ic = nid[i]*Nv*Nv + threadIDx*Nv + threadIDy;
    if (threadIDx < Nv && threadIDy < Nv)
        GI[ic]+=alpha[threadIDx+i*Nv]*btemp[threadIDy]*dark_exp[threadIDx+i*Nv];
     }
}