#ifndef _H_PAFRB_BUFFER
#define _H_PAFRB_BUFFER

/*! \file buffer.cuh
    \brief Defines the main buffer class.

    This is the buffer that is used to aggregate the FFTed data before it is sent to the dedispersion.
    Uses a slightly convoluted version of a ring buffer (the same data chunk is occasionally saved into two places).
*/

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"
#include "obs_time.hpp"

using std::mutex;
using std::vector;

class Buffer {
    private:
        int accumulate_;
        int gpuid_;
        int nochans_;             // number of filterbank channels per time sample
        int nogulps_;
        int nostokes_;             // number of Stokes parameters to keep in the buffer
        int perframe_;
        int fil_saved_;

        mutex buffermutex_;
        mutex statemutex_;

        size_t extrasamples_;           // number of extra time samples required to process the full gulp
        size_t gulpsamples_;            // size of the single gulp
        size_t start_;
        size_t end_;
        size_t totalsamples_;            // total size of the data: #gulps * gulp size + extra samples for dedispersion

        unsigned char **dfilterbank_;
        unsigned char **hdfilterbank_;
        unsigned char **rambuffer_;

        unsigned int typebytes_;

        ObsTime *gulptimes_;

        unsigned int *state_;     // 0 for no data, 1 for data

    protected:

    public:
        Buffer(int id);
        Buffer(int nogulps_u, size_t extrasamples_u, size_t gulpsamples_u, size_t size_u, int id);
        ~Buffer(void);

        unsigned char **GetFilPointer(void) {return this->dfilterbank_;}

        int CheckReadyBuffer(void);

        ObsTime GetTime(int idx);

        template<class BufferType>
        void GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);

        void Allocate(int accumulate, size_t extra, size_t gulp, int filchans, int gulps, int stokes, int perframe, int filbits);
        void Deallocate(void);
        void SendToDisk(int idx, header_f head, std::string telescope, std::string outdir);
        void SendToRam(int idx, cudaStream_t &stream, int host_jump);
        void Update(ObsTime frame_time);
};

template<class BufferType>
void Buffer::GetScaling(int idx, cudaStream_t &stream, float **dmeans, float **drstdevs)
{
    float *dtranspose;
    cudaMalloc((void**)&dtranspose, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(float));
    unsigned char *hdata = new unsigned char[(gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType)];
    cudaCheckError(cudaMemcpy(hdata, hdfilterbank_[0] + (idx - 1) * gulpsamples_ * nochans_ * sizeof(BufferType), (gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType), cudaMemcpyDeviceToHost));

    std::ofstream outdata("nottransposed.dat");
    for (int isamp = 0; isamp < gulpsamples_ + extrasamples_; isamp++) {
        for (int ichan = 0; ichan < nochans_; ichan++) {
            outdata << (float)(reinterpret_cast<BufferType*>(hdata)[isamp * nochans_ + ichan]) << " ";
        }
        outdata << std::endl;
    }
    std::cout << "Saved the non-transposed file" << std::endl;
    outdata.close();
    delete [] hdata;
    float *htranspose = new float[(gulpsamples_ + extrasamples_) * nochans_];
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        TransposeKernel<BufferType, float><<<1,nochans_,0,stream>>>(hdfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_ * sizeof(BufferType), dtranspose, nochans_, gulpsamples_ + extrasamples_);
        if (istoke == 0)
            cudaCheckError(cudaMemcpy(htranspose, dtranspose, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(float), cudaMemcpyDeviceToHost));
        ScaleFactorsKernel<<<1,nochans_,0,stream>>>(dtranspose, dmeans, drstdevs, nochans_, gulpsamples_ + extrasamples_, istoke);
    }
    cudaFree(dtranspose);
    std::ofstream outfile("transposed.dat");
    for (int ichan = 0; ichan < nochans_; ichan++) {
        for (int isamp = 0; isamp < gulpsamples_ + extrasamples_; isamp++) {
            outfile << htranspose[ichan * (gulpsamples_ + extrasamples_) + isamp] << " ";
        }
        outfile << std::endl;
    }
    std::cout << "Saved the transpose file..." << std::endl;
    outfile.close();
    delete [] htranspose;
    // need this so I don't save this buffer
    statemutex_.lock();
    state_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();
}


#endif
