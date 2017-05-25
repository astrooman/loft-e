#include <stdio.h>

#include <kernels.cuh>

// __restrict__ tells the compiler there is no memory overlap

__device__ float fftfactor = 1.0/32.0 * 1.0/32.0;

// TODO: have this change depending on the unpack factor
__constant__ unsigned char kMask[] = {0x03, 0x0C, 0x30, 0xC0};

__global__ void UnpackKernel(unsigned char __restrict__ **in, float __restrict__ **out, int pols, int perthread, int rem, size_t samples, int unpack)
{
    int idx = blockIdx.x * blockDim.x * perthread + threadIdx.x;
    int skip = blockDim.x;

    if (idx < blockDim.x * gridDim.x - (blockDim.x - rem)) {
        if ((blockIdx.x == (gridDim.x -1)) && (rem != 0)) {
            skip = rem;
        }

        for (int ipol = 0; ipol < pols; ipol++) {
            for (int isamp = 0; isamp < perthread; isamp++) {
                for (int ipack = 0; ipack < unpack; ipack++) {
                    out[ipol][(idx + isamp * skip) * unpack + ipack] = static_cast<float>(static_cast<short>((in[ipol][idx + isamp * skip] & kMask[ipack]) >> ( 2 * ipack)));
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
    unsigned int inidx = 0;
    unsigned int outidx = 0;
    unsigned int filtimeidx = 0;


    for (int ichunk = 0; ichunk < outsampperblock; ichunk++) {
        filtimeidx = framet * perframe + blockIdx.x * outsampperblock + ichunk;
        outidx = filtimeidx * nchans + threadIdx.x;
        for (int isamp = 0; isamp < avgtime; isamp++) {
            for (int ifreq = 0; ifreq < avgfreq; ifreq++) {
                inidx = inskip + blockIdx.x * avgtime * nchans * outsampperblock + ichunk * nchans * avgtime + isamp * nchans + threadIdx.x * avgfreq + ifreq;
                out[0][outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                out[1][outidx] += in[0][inidx].x * in[0][inidx].x + in[0][inidx].y * in[0][inidx].y + in[1][inidx].x * in[1][inidx].x + in[1][inidx].y * in[1][inidx].y;
                out[2][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].x + 2.0f * in[0][inidx].y * in[1][inidx].y;
                out[3][outidx] += 2.0f * in[0][inidx].x * in[1][inidx].y + 2.0f * in[0][inidx].y * in[1][inidx].x;
            }
        }
        // TODO: save the data in two places in memory
        out[0][outidx] = (out[0][outidx] - means[0][threadIdx.x]) / stdevs[0][threadIdx.x] * 32.0 + 64.0;
        out[1][outidx] = (out[1][outidx] - means[1][threadIdx.x]) / stdevs[1][threadIdx.x] * 32.0 + 64.0;
        out[2][outidx] = (out[2][outidx] - means[2][threadIdx.x]) / stdevs[2][threadIdx.x] * 32.0 + 64.0;
        out[3][outidx] = (out[3][outidx] - means[3][threadIdx.x]) / stdevs[3][threadIdx.x] * 32.0 + 64.0;

        if (filtimeidx < extra) {
            out[0][outidx + nogulps * gulpsize * nchans] = out[0][outidx];
            out[1][outidx + nogulps * gulpsize * nchans] = out[1][outidx];
            out[2][outidx + no gulps * gulpsize * nchans] = out[2][outidx];
            out[3][outidx + nogulps * gulpsize * nchans] = out[3][outidx];
        }
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
__global__ void TransposeKernel(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes) {

    // very horrible implementation or matrix transpose
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * ntimes;
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        out[start + tsamp] = in[idx + tsamp * nchans];
    }
}

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
