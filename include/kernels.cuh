#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void UnpackKernel(unsigned char **in, float **out, int pols, int perthread, int rem, size_t samples, int unpack);

__global__ void PowerKernel(cufftComplex **in, float **out);

//__global__ void PowerScaleKernel(cufftComplex **in, unsigned char **out, float **means, float **stdevs, int avgfreq, int avgtime, int nchans);

__global__ void PowerScaleKernel(cufftComplex **in, unsigned char **out, float **means, float **stdevs,
                                    int avgfreq, int avgtime, int nchans, int outsampperblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet,
                                    unsigned int perframe);

__global__ void ScaleKernel(float **in, unsigned char **out);

__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort);

__global__ void addchannel2(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize, short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc);

__global__ void addchanscale(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize, short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc, float **means, float **rstdevs);

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump);

__global__ void powertime(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort);

__global__ void powertime2(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort, unsigned int acc);

__global__ void initscalefactors(float **means, float **rstdevs, int stokes);

__global__ void transpose(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes);

__global__ void scale_factors(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param);

#endif
