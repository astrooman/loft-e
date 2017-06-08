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

        ObsTime *gulptimes_;

        unsigned int *state_;     // 0 for no data, 1 for data

    protected:

    public:
        Buffer(int id);
        Buffer(int nogulps_u, size_t extrasamples_u, size_t gulpsamples_u, size_t size_u, int id);
        ~Buffer(void);

        BufferType **GetFilPointer(void) {return this->dfilterbank_;};

        int CheckReadyBuffer(void);

        ObsTime GetTime(int idx);

        template<class BufferType>
        void Allocate(int accumulate, size_t extra, size_t gulp, int filchans, int gulps, int stokes, int perframe);
        void Deallocate(void);
        void SendToDisk(int idx, header_f head, std::string telescope, std::string outdir);
        void SendToRam(int idx, cudaStream_t &stream, int host_jump);
        void GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);
        void Update(ObsTime frame_time);
};

template<class BufferType>
Buffer<BufferType>::Buffer(int id) : gpuid_(id) {
    cudaSetDevice(gpuid_);
    start_ = 0;
    end_ = 0;
}

template<class BufferType>
Buffer<BufferType>::Buffer(int nogulps_u, size_t extrasamples_u, size_t gulpsamples_u, size_t size_u, int id) : extrasamples_(extrasamples_u),
                                                                                gulpsamples_(gulpsamples_u),
                                                                                nogulps_(nogulps_u),
                                                                                totalsamples_(size_u),
                                                                                gpuid_(id) {
    start_ = 0;
    end_ = 0;
    state_ = new unsigned int[(int)totalsamples_];
    std::fill(state_, state_ + totalsamples_, 0);
}

template<class BufferType>
Buffer<BufferType>::~Buffer() {
    end_ = 0;
}

template<class BufferType>
void Buffer<BufferType>::Allocate(int accumulate, size_t extra, size_t gulp, int filchans, int gulps, int stokes, int perframe) {
    fil_saved_ = 0;
    accumulate_ = accumulate;
    extrasamples_ = extra;
    gulpsamples_ = gulp;
    nochans_ = filchans;
    nogulps_ = gulps;
    nostokes_ = stokes;
    perframe_ = perframe;
    // size for a single Stokes parameter
    totalsamples_ = nogulps_ * gulpsamples_ + extrasamples_;

    std::cout << totalsamples_ << std::endl;
    std::cout << totalsamples_ * nochans_ << std::endl;

    gulptimes_ = new ObsTime[nogulps_];
    hdfilterbank_ = new unsigned char*[nostokes_];
    state_ = new unsigned int[(int)totalsamples_];
    cudaCheckError(cudaHostAlloc((void**)&rambuffer_, nostokes_ * sizeof(unsigned char*), cudaHostAllocDefault));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&hdfilterbank_[istoke], totalsamples_ * nochans_ * sizeof(BufferType)));
        cudaCheckError(cudaHostAlloc((void**)&rambuffer_[istoke], 2 * (gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType), cudaHostAllocDefault));
        std::cout << "Stokes " << istoke << " done" << std::endl;
    }

    cudaCheckError(cudaMalloc((void**)&dfilterbank_, nostokes_ * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dfilterbank_, hdfilterbank_, nostokes_ * sizeof(unsigned char*), cudaMemcpyHostToDevice));
    std::cout << "Other memory done" << std::endl;
}

template<class BufferType>
void Buffer<BufferType>::Deallocate(void) {

    cudaCheckError(cudaFree(dfilterbank_));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFreeHost(rambuffer_[istoke]));
        cudaCheckError(cudaFree(hdfilterbank_[istoke]));
    }

    cudaCheckError(cudaFreeHost(rambuffer_));

    delete [] hdfilterbank_;
    delete [] gulptimes_;
}

template<class BufferType>
void Buffer<BufferType>::SendToDisk(int idx, header_f header, std::string telescope, std::string outdir) {
    SaveFilterbank<BufferType>(rambuffer_, gulpsamples_ + extrasamples_, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType) * idx, header, nostokes_, fil_saved_, telescope, outdir);
    fil_saved_++;
    // need info from the telescope
}

template<class BufferType>
int Buffer<BufferType>::CheckReadyBuffer(void) {
    std::lock_guard<mutex> addguard(statemutex_);
    // for now check only the last position for the gulp
    for (int igulp = 0; igulp < nogulps_; igulp++) {
        if (state_[(igulp + 1) * gulpsamples_ + extrasamples_ - 1] == 1)
            return (igulp + 1);
    }
    return 0;
}

template<class BufferType>
void Buffer<BufferType>::GetScaling(int idx, cudaStream_t &stream, float **dmeans, float **drstdevs)
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


template<class BufferType>
void Buffer<BufferType>::SendToRam(int idx, cudaStream_t &stream, int host_jump) {
    // which half of the RAM buffer we are saving into
    host_jump *= (gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType);
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMemcpyAsync(rambuffer_[istoke] + host_jump, hdfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_ * sizeof(BufferType), (gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType), cudaMemcpyDeviceToHost, stream));
        std::cout << "Sent stokes " << istoke << std::endl;
    }
    cudaStreamSynchronize(stream);
    std::cout << "Sent to RAM... " << std::endl;
    statemutex_.lock();
    // HACK: the call below is wrong - restarts the whole sample state
    //std::fill(sample_state, sample_state + totsize, 0);
    state_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();
}


/* template<class BufferType>
void Buffer<BufferType>::Update(ObsTime frametime) {
    std::lock_guard<mutex> addguard(statemutex_);
    int framet = frametime.framet;
    int index = 0;
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
    for (int isamp = 0; isamp < accumulate_; isamp++) {
        index = framet % (nogulps_ * gulpsamples_);
        if ((index % gulpsamples_) == 0)
            gulptimes_[index / gulpsamples_] = frametime;
        state_[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        //std::cout.flush();
        // second condition is to avoid sending the second buffer when the very fisrt buffer is being filled
        if ((index < extrasamples_) && (framet > extrasamples_)) {
            state_[index + nogulps_ * gulpsamples_] = 1;
        }
        framet++;
    }
} */

template<class BufferType>
void Buffer<BufferType>::Update(ObsTime frametime) {
    std::lock_guard<mutex> addguard(statemutex_);
    int framet = frametime.framet;
    int filtime = frametime.framet * perframe_;
    int index = 0;
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
    for (int isamp = 0; isamp < accumulate_ * perframe_; isamp++) {
        index = filtime % (nogulps_ * gulpsamples_);
        if ((index % gulpsamples_) == 0)
            gulptimes_[index / gulpsamples_] = frametime;
        state_[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        //std::cout.flush();
        // second condition is to avoid sending the second buffer when the very fisrt buffer is being filled
        if ((index < extrasamples_) && (framet > extrasamples_)) {
            state_[index + nogulps_ * gulpsamples_] = 1;
        }
        filtime++;
    }
}

template<class BufferType>
ObsTime Buffer<BufferType>::GetTime(int idx) {
    return gulptimes_[idx];
}

#endif
