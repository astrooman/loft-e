#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void rearrange(cudaTextureObject_t texObj, cufftComplex * __restrict__ out);

__global__ void rearrange2(cudaTextureObject_t texObj, cufftComplex * __restrict__ out, unsigned int acc);

__global__ void addtime(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factort);

__global__ void addchannel(float *in, float *out, unsigned int jumpin, unsigned int jumpout, unsigned int factorc);

__global__ void addchannel2(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize, short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc);

__global__ void addchanscale(float* __restrict__ in, float** __restrict__ out, short nchans, size_t gulp, size_t totsize, short gulpno, unsigned int jumpin, unsigned int factorc, unsigned int framet, unsigned int acc, float **means, float **rstdevs);

__global__ void powerscale(cufftComplex *in, float *out, unsigned int jump);

__global__ void powertime(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort);

__global__ void powertime2(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort, unsigned int acc);

__global__ void initscalefactors(float **means, float **rstdevs, int stokes);

__global__ void transpose(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes);

__global__ void scale_factors(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param);

__global__ void bandpass();
#endif
