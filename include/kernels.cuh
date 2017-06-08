#ifndef _H_PAFRB_KERNELS
#define _H_PAFRB_KERNELS

#include <cufft.h>
__global__ void UnpackKernel(unsigned char **in, float **out, int pols, int perthread, int rem, size_t samples, int unpack);

__global__ void PowerScaleKernel(cufftComplex **in, unsigned char **out, float **means, float **stdevs,
                                    int avgfreq, int avgtime, int nchans, int outsampperblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet,
                                    unsigned int perframe);

__global__ void ScaleFactorsInitKernel(float **means, float **rstdevs, int stokes);

template<class BufferType, class OutType>
__global__ void TransposeKernel(unsigned char* __restrict__ in, OutType* __restrict__ out, unsigned int nchans, unsigned int ntimes) {
    // very horrible implementation or matrix transpose
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * ntimes;
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        out[start + tsamp] = (OutType)(reinterpret_cast<BufferType*>(in))[idx + tsamp * nchans];
    }
}


template<class OutType>
__global__ void PowerScaleKernel(cufftComplex **in, unsigned char **out, float **means, float **stdevs,
                                    int avgfreq, int avgtime, int nchans, int outsampperblock,
                                    int inskip, int nogulps, int gulpsize, int extra, unsigned int framet,
                                    unsigned int perframe)
{
    // NOTE: nchans is the number of frequency channels AFTER the averaging
    unsigned int inidx = 0;
    unsigned int outidx = 0;
    unsigned int filtimeidx = 0;
    unsigned int filfullidx = 0;

    for (int ichunk = 0; ichunk < outsampperblock; ichunk++) {
        filtimeidx = framet * perframe + blockIdx.x * outsampperblock + ichunk;
        filfullidx = (filtimeidx % (nogulps * gulpsize)) * nchans;
        outidx = filfullidx + threadIdx.x;
        for (int isamp = 0; isamp < avgtime; isamp++) {
            for (int ifreq = 0; ifreq < avgfreq; ifreq++) {
                //inidx = inskip + blockIdx.x * avgtime * nchans * outsampperblock + ichunk * nchans * avgtime + isamp * nchans + threadIdx.x * avgfreq + ifreq;
		//inidx = inskip + blockIdx.x * outsampperblock * avgtime * nchans * avgfreq + ichunk * avgtime * nchans * avgfreq + isamp * nchans * avgfreq  + threadIdx.x * avgfreq + ifreq;
                inidx = inskip + blockIdx.x * outsampperblock * avgtime * (nchans + 1) * avgfreq + ichunk * avgtime * (nchans + 1) * avgfreq + isamp * (nchans + 1) * avgfreq  + threadIdx.x * avgfreq + ifreq + 1;
                (reinterpret_cast<OutType*>(out[0]))[outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                (reinterpret_cast<OutType*>(out[1]))[outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                (reinterpret_cast<OutType*>(out[2]))[outidx] += 2.0f * in[0][inidx].x * in[1][inidx].x + 2.0f * in[0][inidx].y * in[1][inidx].y;
                (reinterpret_cast<OutType*>(out[3]))[outidx] += 2.0f * in[0][inidx].x * in[1][inidx].y + 2.0f * in[0][inidx].y * in[1][inidx].x;
            }
        }
        //out[0][outidx] = (out[0][outidx] * fftfactor - means[0][threadIdx.x]) / stdevs[0][threadIdx.x] * 32.0 + 64.0;
        //out[1][outidx] = (out[1][outidx] * fftfactor - means[1][threadIdx.x]) / stdevs[1][threadIdx.x] * 32.0 + 64.0;
        //out[2][outidx] = (out[2][outidx] * fftfactor - means[2][threadIdx.x]) / stdevs[2][threadIdx.x] * 32.0 + 64.0;
        //out[3][outidx] = (out[3][outidx] * fftfactor - means[3][threadIdx.x]) / stdevs[3][threadIdx.x] * 32.0 + 64.0;

        if (filfullidx < extra) {
            (reinterpret_cast<OutType*>(out[0]))[outidx + nogulps * gulpsize * nchans] = (reinterpret_cast<OutType*>(out[0]))[outidx];
            (reinterpret_cast<OutType*>(out[1]))[outidx + nogulps * gulpsize * nchans] = (reinterpret_cast<OutType*>(out[1]))[outidx];
            (reinterpret_cast<OutType*>(out[2]))[outidx + nogulps * gulpsize * nchans] = (reinterpret_cast<OutType*>(out[2]))[outidx];
            (reinterpret_cast<OutType*>(out[3]))[outidx + nogulps * gulpsize * nchans] = (reinterpret_cast<OutType*>(out[3]))[outidx];
        }
    }
}

__global__ void ScaleFactorsKernel(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param);

#endif
