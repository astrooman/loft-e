kMask#include <stdio.h>

#include <kernels.cuh>

#define MIDAVG 4
#define NACCUMULATE 4000
#define NACCPROCESS 1000
#define PERBLOCK 625
#define TIMEAVG 16
#define TIMESCALE 0.125
#define FFTOUT 257
#define FFTUSE 256
#define VDIFLEN 8000
// NOTE: The number of time samples that come out of the fully averaged time series
#define AVGOUTSAMPS 15625

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

__global__ void UnpackKernelSmall(unsigned char *in, float *out, size_t samples) {

    // NOTE: Each thread in the block processes 625 samples
    int idx = blockIdx.x * blockDim.x * PERBLOCK + threadIdx.x;
    int tmod = threadIdx.x % 4;
    int tdiv = threadIdx.x >> 2;
    int outidx2 = 0;
    unsigned char inval = 0;

    __shared__ unsigned char incoming[1024];

    int outidx = blockIdx.x * blockDim.x * PERBLOCK * 4;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {
        if (idx < samples) {
            incoming[threadIdx.x] = in[idx];
            __syncthreads();
            outidx2 = outidx + threadIdx.x;
            for (int ichunk = 0; ichunk < 4; ++ichunk) {
                int inidx = tdiv + ichunk * 256;
                inval = incoming[inidx];
                out[outidx2] = static_cast<float>(static_cast<short>(((inval & kMask[tmod]) >> (2 * tmod))));
                outidx2 += 1024;
            }
        }
        idx += blockDim.x;
        outidx += blockDim.x * 4;
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

__global__ void PowerKernelSmall(cufftComplex *in, float *out, int ichunk) {
    int band = threadIdx.x / 256;
    int threadinband = threadIdx.x % 256;
    float sum = 0.0f;
    cufftComplex polval;
    int inpolskip = NACCPROCESS * VDIFLEN * 4 / (2 * FFTUSE) * FFTOUT;
    int outpolskip = NACCUMULATE * VDIFLEN * 4 / (2 * FFTUSE) / MIDAVG * FFTUSE;

    // NOTE: The chunk of NACCPROCESS = 1000 will result in 62500 time samples after 512-point FFT
    // After 4-point averaging this gives 16250 time samples.
    // With all the chunks in place, the buffer with have 62500 time samples again, but this time averaged

    // NOTE: polskip skips one polarisation in the band (2 * polskip to skip the whole band)
    int inidx = band * 2 * inpolskip + blockIdx.x * PERBLOCK * MIDAVG * FFTOUT + threadinband + 1;
    // NOTE: outpolskip skips all the time samples in the full buffer
    // NOTE: outpolskip / 4 skips only the time samples relevant for the given data chunk
    int outidx = band * outpolskip + ichunk * outpolskip / MIDAVG + blockIdx.x * PERBLOCK * FFTUSE + threadinband;


    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {
        for (int iavg = 0; iavg < MIDAVG; ++iavg) {
            polval = in[inidx + iavg * FFTOUT];
            sum += polval.x * polval.x + polval.y * polval.y;
            polval = in[inidx + polskip + iavg * FFTOUT];
            sum += polval.x * polval.x + polval.y * polval.y;
        }
        out[outidx] = sum;
        sum = 0.0f;
        inidx += MIDAVG * FFTOUT;
        outidx += FFTUSE;
    }
}

__global__ void PowerAvgKernel(float *in, float *out) {
    int band = threadIdx.x / 256;
    int threadinband = threadIdx.x % 256;
    float sum = 0.0f;

    int bandskip = NACCUMULATE * VDIFLEN * 4 / (2 * FFTUSE) / MIDAVG * FFTUSE;

    // NOTE: As we average by 16 in total, initial and final averages are both 4
    int inidx = band * bandskip + blockIdx.x * PERBLOCK * MIDAVG * FFTUSE + threadinband;
    // NOTE: We only have to skip a quarted of the incoming band samples for the output as we average by 4
    // NOTE: FFTUSE - threadinband - 1 to invert the frequency ordering with the top frequency first
    int outidx = band * bandskip  / MIDAVG + blockIdx.x * PERBLOCK * FFTUSE + FFTUSE - threadinband - 1;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {
        for (int iavg = 0; iavg < MIDAVG; ++iavg) {
            sum += in[inidx + iavg * FFTUSE];
        }
        out[outidx] = sum;
        sum = 0.0f;
        inidx += MIDAVG * FFTUSE;
        outidx += FFTUSE;

    }
}

__global__ PowerAvgScaleKernel(float *in, float *means, float *stdevs, unsigned char *out, int nogulps, int gulpsize, int extra, unsigned int framet, unsigned int skipchans) {
    int band = threadIdx.x / 256;
    int threadinband = threadIdx.x % 256;
    float sum = 0.0f;
    int scaled = 0;

    unsigned int inidx = band * bandskip + blockIdx.x * PERBLOCK * MIDAVG * FFTUSE + threadinband;
    unsigned int outidx;
    unsigned int filbufidx;

    // NOTE: A chunk of 4000 frames will provide 15625 time samples
    // Each block provides 625 output time samples
    unsigned int filtime = framet / NACCUMULATE * AVGOUTSAMPS + blockIdx.x * PERBLOCK;

    for (int isamp = 0; isamp < PERBLOCK; ++isamp) {
        for (int iavg = 0; iavg < MIDAVG; ++iavg) {
            sum += in[inidx + iavg * FFTUSE];
        }

        filbufidx = filtime % (nogulps * gulpsize);
        outidx = filbufidx * (2 * FFTUSE + skipchans) + band * skipchans + FFTUSE - threadIdx.x - 1;

        scaled = __float2int_ru((sum - means[FFTUSE - threadIdx.x - 1]) / stdevs[FFTUSE - threadIdx.x - 1] * 32.0f + 128.0f);
        if (scaled > 255) {
            scaled = 255;
        } else if (scale < 0) {
            scaled = 0;
        }

        out[outidx] = (unsigned char)scaled;
        if (filbufidx < extra) {
            out[outidx + nogulps * gulpsize * (2 * FFTUSE + skipchans)] = (unsigned char)scaled;
        }

        sum = 0.0f;
        filtime++;
        inidx += MIDAVG * FFTUSE;
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
