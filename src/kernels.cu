#include <stdio.h>

#include <kernels.cuh>

#define ACC 1024
#define PERBLOCK 625
#define TIMEAVG 8
#define TIMESCALE 0.125
#define FFTOUT 513
#define FFTUSE 512

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

// NOTE: Does not do any frequency averaging
// NOTE: Outputs only the total intensity and no other Stokes parameters
__global__ void PowerScaleKernelOpt(cufftComplex **in, unsigned char **out, int nogulps, int gulpsize, int extra, unsigned int framet) {
    // NOTE: framet should start at 0 and increase by accumulate every time this kernel is called
    // NOTE: REALLY make sure it starts at 0
    // NOTE: I'M SERIOUS - FRAME TIME CALCULATIONS ARE BASED ON THIS ASSUMPTION
    unsigned int filtime = framet / ACC * gridDim.x * PERBLOCK + blockIdx.x * PERBLOCK;
    unsigned int filidx;
    unsigned int outidx;
    int inidx = blockIdx.x * PERBLOCK * TIMEAVG * FFTOUT + threadIdx.x + 1;

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

        filidx = filtime % (nogulps * gulpsize);
        outidx = filidx * FFTUSE + threadIdx.x;

        outvalue *= TIMESCALE;

        out[0][outidx] = outvalue;
        // NOTE: Save to the extra part of the buffer
        if (filidx < extra) {
            out[0][outidx + nogulps * gulpsize * FFTUSE] = outvalue;
        }
        inidx += FFTOUT * TIMEAVG;
        filtime++;
        outvalue = 0.0;
    }
}

// Initialise the scale factors
// Use this instead of memset
// NOTE: Memset is slower than custom kernels and not safe for anything else than int
__global__ void ScaleFactorsInitKernel(float **means, float **rstdevs, int stokes) {
    // the scaling is (in - mean) * rstdev + 64.0f
    // and I want to get the original in back in the first running
    // will therefore set the mean to 64.0f and rstdev to 1.0f

    // each thread responsible for one channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int ii = 0; ii < stokes; ii++) {
        means[ii][idx] = 64.0f;
        rstdevs[ii][idx] = 1.0f;
    }
}

// NOTE: Filterbank data saved in the format t1c1,t1c2,t1c3,...
// Need to transpose to t1c1,t2c1,t3c1,... for easy and efficient scaling kernel
/*__global__ void TransposeKernel(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes) {

    // very horrible implementation or matrix transpose
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * ntimes;
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        out[start + tsamp] = in[idx + tsamp * nchans];
    }
}*/

__global__ void ScaleFactorsKernel(float *in, float **means, float **rstdevs, unsigned int nchans, unsigned int ntimes, int param) {
    // calculates mean and standard deviation in every channel
    // assumes the data has been transposed

    // for now have one thread per frequency channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float mean;
    float variance;

    float ntrec = 1.0f / (float)ntimes;
    float ntrec1 = 1.0f / (float)(ntimes - 1.0f);

    unsigned int start = idx * ntimes;
    mean = 0.0f;
    variance = 0.0;
    // two-pass solution for now
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        mean += in[start + tsamp] * ntrec;
    }
    means[param][idx] = mean;

    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        variance += (in[start + tsamp] - mean) * (in[start + tsamp] - mean);
    }
    variance *= ntrec1;
    // reciprocal of standard deviation
    // multiplied by the desired standard deviation of the scaled data
    // reduces the number of operations that have to be done on the GPU
    rstdevs[param][idx] = rsqrtf(variance) * 32.0f;
    // to avoid inf when there is no data in the channel
    if (means[param][idx] == 0)
        rstdevs[param][idx] = 0;
}
