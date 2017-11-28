#include <stdio.h>

#include <kernels.cuh>

#define NACCUMULATE 4000
#define PERBLOCK 625
#define TIMEAVG 16
#define TIMESCALE 0.125
#define FFTOUT 257
#define FFTUSE 256

// __restrict__ tells the compiler there is no memory overlap

//__device__ float fftfactor = 1.0/256.0 * 1.0/256.0;
__device__ float fftfactor = 1.0;

// TODO: have this change depending on the unpack factor
__constant__ unsigned char kMask[] = {0x03, 0x0C, 0x30, 0xC0};

__global__ void UnpackKernel(unsigned char **in, float **out, int nopols, int bytesperthread, int rem, size_t samples, int unpack)
{
    int idx = blockIdx.x * blockDim.x * bytesperthread + threadIdx.x * bytesperthread;

    if (idx < samples) {
        for (int ipol = 0; ipol < nopols; ipol++) {
            for (int ibyte = 0; ibyte < bytesperthread; ibyte++) {
                for (int isamp = 0; isamp < unpack; isamp++) {
                    out[ipol][idx * unpack + ibyte * unpack + isamp] = static_cast<float>(static_cast<short>((in[ipol][idx + ibyte] & kMask[isamp]) >> ( 2 * isamp)));
                }
            }
        }
    }
}

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
                out[0][outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                out[1][outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                out[2][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].x + 2.0f * in[0][inidx].y * in[1][inidx].y;
                out[3][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].y + 2.0f * in[0][inidx].y * in[1][inidx].x;
            }
        }
        // TODO: save the data in two places in memory
        //out[0][outidx] = (out[0][outidx] * fftfactor - means[0][threadIdx.x]) / stdevs[0][threadIdx.x] * 32.0 + 64.0;
        //out[1][outidx] = (out[1][outidx] * fftfactor - means[1][threadIdx.x]) / stdevs[1][threadIdx.x] * 32.0 + 64.0;
        //out[2][outidx] = (out[2][outidx] * fftfactor - means[2][threadIdx.x]) / stdevs[2][threadIdx.x] * 32.0 + 64.0;
        //out[3][outidx] = (out[3][outidx] * fftfactor - means[3][threadIdx.x]) / stdevs[3][threadIdx.x] * 32.0 + 64.0;

        if (filfullidx < extra) {
            out[0][outidx + nogulps * gulpsize * nchans] = out[0][outidx];
            out[1][outidx + nogulps * gulpsize * nchans] = out[1][outidx];
            out[2][outidx + nogulps * gulpsize * nchans] = out[2][outidx];
            out[3][outidx + nogulps * gulpsize * nchans] = out[3][outidx];
        }
    }
}

__global__ void UnpackKernelOpt(unsigned char **in, float **out, size_t samples) {

    // NOTE: Each thread in the block processes 625 samples
    int idx = blockIdx.x * blockDim.x * PERBLOCK + threadIdx.x;
    int tmod = threadIdx.x % 4;

    // NOTE: Each thread can store one value
    __shared__ unsigned char incoming[1024];

    int outidx = blockIdx.x * blockDim.x * PERBLOCK * 4;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {
        if (idx < samples) {
            for (int ipol = 0; ipol < 2; ++ipol) {
                incoming[threadIdx.x] = in[ipol][idx];
                __syncthreads();
                int outidx2 = outidx + threadIdx.x;
                for (int ichunk = 0; ichunk < 4; ++ichunk) {
                    int inidx = threadIdx.x / 4 + ichunk * 256;
                    unsigned char inval = incoming[inidx];
                    out[ipol][outidx2] = static_cast<float>(static_cast<short>(((inval & kMask[tmod]) >> (2 * tmod))));
                    outidx2 += 1024;
                }
            }
        }
        idx += blockDim.x;
        outidx += blockDim.x * 4;
    }
}

__global__ void PowerKernelOpt(cufftComplex **in, float *out) {
    // NOTE: framet should start at 0 and increase by accumulate every time this kernel is called
    // NOTE: REALLY make sure it starts at 0
    // NOTE: I'M SERIOUS - FRAME TIME CALCULATIONS ARE BASED ON THIS ASSUMPTION
    unsigned int outidx = blockIdx.x * PERBLOCK * FFTUSE + FFTUSE - threadIdx.x - 1;
    unsigned int inidx = blockIdx.x * PERBLOCK * TIMEAVG * FFTOUT + threadIdx.x + 1;

    float outvalue = 0.0f;
    cufftComplex polval;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {
        // NOTE: Read the data from the incoming array
        for (int ipol = 0; ipol < 2; ++ipol) {
            for (int iavg = 0; iavg < TIMEAVG; iavg++) {
                polval = in[ipol][inidx + iavg * FFTOUT];
                outvalue += polval.x * polval.x + polval.y * polval.y;
            }

        }
        // outidx = blockIdx.x * PERBLOCK * FFTUSE + isamp * FFTUSE + FFTUSE - threadIdx.x - 1;
        outvalue *= TIMESCALE;
        out[outidx] = outvalue;

        inidx += FFTOUT * TIMEAVG;
        outidx += FFTUSE;
        outvalue = 0.0;
    }
}

// NOTE: Does not do any frequency averaging
// NOTE: Outputs only the total intensity and no other Stokes parameters
__global__ void PowerScaleKernelOpt(cufftComplex **in, float *means, float *stdevs, unsigned char **out, int nogulps, int gulpsize, int extra, unsigned int framet) {
    // NOTE: framet should start at 0 and increase by accumulate every time this kernel is called
    // NOTE: REALLY make sure it starts at 0
    // NOTE: I'M SERIOUS - FRAME TIME CALCULATIONS ARE BASED ON THIS ASSUMPTION
    unsigned int filtime = framet / NACCUMULATE * gridDim.x * PERBLOCK + blockIdx.x * PERBLOCK;
    unsigned int filidx;
    unsigned int outidx;
    int inidx = blockIdx.x * PERBLOCK * TIMEAVG * FFTOUT + threadIdx.x + 1;

    float outvalue = 0.0f;
    cufftComplex polval;

    int scaled = 0;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {

        // NOTE: Read the data from the incoming array
        for (int ipol = 0; ipol < 2; ++ipol) {
            for (int iavg = 0; iavg < TIMEAVG; iavg++) {
                polval = in[ipol][inidx + iavg * FFTOUT];
                outvalue += polval.x * polval.x + polval.y * polval.y;
            }

        }

        filidx = filtime % (nogulps * gulpsize);
        outidx = filidx * FFTUSE + FFTUSE - threadIdx.x - 1;
        outvalue *= TIMESCALE;

        scaled = __float2int_ru((outvalue - means[FFTUSE - threadIdx.x - 1]) / stdevs[FFTUSE - threadIdx.x - 1] * 32.0f + 128.0f);

        if (scaled > 255) {
            scaled = 255;
        } else if (scaled < 0) {
            scaled = 0;
        }

        out[0][outidx] = (unsigned char)scaled;
        // NOTE: Save to the extra part of the buffer
        if (filidx < extra) {
            out[0][outidx + nogulps * gulpsize * FFTUSE] = (unsigned char)scaled;
        }
        inidx += FFTOUT * TIMEAVG;
        filtime++;
        outvalue = 0.0;
    }
}

// NOTE: Uses a simple implementation of Chan's algorithm
__global__ void GetScaleFactorsKernel(float *in, float *means, float *stdevs, float *factors, size_t processed) {

    // NOTE: Filterbank file format coming in
    //float mean = indata[threadIdx.x];
    float mean = 0.0f;
    // NOTE: Depending whether I save STD or VAR at the end of every run
    // float estd = stdev[threadIdx.x];
    float estd = stdevs[threadIdx.x] * stdevs[threadIdx.x] * (processed - 1.0f);
    float oldmean = means[threadIdx.x];

    //float estd = 0.0f;
    //float oldmean = 0.0;

    float val = 0.0f;
    float diff = 0.0;

    for (int isamp = 0; isamp < 15625; ++isamp) {
        val = in[isamp * blockDim.x + threadIdx.x];
        diff = val - oldmean;
        mean = oldmean + diff * factors[processed + isamp + 1];
        estd += diff * (val - mean);
        oldmean = mean;
    }
    means[threadIdx.x] = mean;
    stdevs[threadIdx.x] = sqrtf(estd / (float)(processed + 15625 - 1.0f));
    // stdev[threadIdx.x] = estd;
}
