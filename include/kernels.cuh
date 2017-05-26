#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void UnpackKernel(unsigned char **in, float **out, int pols, int perthread, int rem, size_t samples, int unpack);

__global__ void PowerScaleKernel(cufftComplex **in, unsigned char **out, float **means, float **stdevs,
                                    int avgfreq, int avgtime, int nchans, int outsampperblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet,
                                    unsigned int perframe);

__global__ void ScaleFactorsInitKernel(float **means, float **rstdevs, int stokes);

template<class InType, class OutType>
__global__ void TransposeKernel(InType* __restrict__ in, OutType* __restrict__ out, unsigned int nchans, unsigned int ntimes) {
    // very horrible implementation or matrix transpose
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * ntimes;
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        out[start + tsamp] = (OutType)in[idx + tsamp * nchans];
    }
}


__global__ void ScaleFactorsKernel(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param);

#endif
