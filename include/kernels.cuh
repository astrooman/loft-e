#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void UnpackKernel(unsigned char __restrict__ **in, float __restrict__ **out, int pols, int perthread, int rem, size_t samples, int unpack);

__global__ void PowerScaleKernel(cufftComplex **in, unsigned char **out, float **means, float **stdevs,
                                    int avgfreq, int avgtime, int nchans, int outsampperblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet,
                                    unsigned int perframe);

__global__ void ScaleFactorsInitKernel(float **means, float **rstdevs, int stokes);

__global__ void TransposeKernel(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes);

__global__ void ScaleFactorsKernel(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param);

#endif
