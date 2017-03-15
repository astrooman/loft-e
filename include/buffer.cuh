#ifndef _H_PAFRB_BUFFER
#define _H_PAFRB_BUFFER

/*! \file buffer.cuh
    \brief Defines the main buffer class.

    This is the buffer that is used to aggregate the FFTed data before it is sent to the dedispersion.
    Uses a slightly convoluted version of a ring buffer (the same data chunk is occasionally saved into two places).
*/

#include <algorithm>
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

template <class BufferType>
class Buffer
{
    private:
        vector<thrust::device_vector<BufferType>> d_filterbank_;              // stores different Stoke parameters
        vector<thrust::host_vector<BufferType>> h_filterbank_;                // stores Stokes parameters in the RAM buffer



        int accumulate_;
        int gpuid_;
        int nochans_;             // number of filterbank channels per time sample
        int nogulps_;
        int nostokes_;             // number of Stokes parameters to keep in the buffer
        int fil_saved_;

        mutex buffermutex_;
        mutex statemutex_;

        size_t extrasamples_;           // number of extra time samples required to process the full gulp
        size_t gulpsamples_;            // size of the single gulp
        size_t start_;
        size_t end_;
        size_t totalsamples_;            // total size of the data: #gulps * gulp size + extra samples for dedispersion

        BufferType **dfilterbank_;
        BufferType **hdfilterbank_;
        BufferType **rambuffer_;

        ObsTime *gulptimes_;

        unsigned int *state_;     // 0 for no data, 1 for data

    protected:

    public:
        Buffer(int id);
        Buffer(int nogulps_u, size_t extrasamples_u, size_t gulpsamples_u, size_t size_u, int id);
        ~Buffer(void);

        void Allocate(int accumulate, size_t extra, size_t gulp, int filchans, int gulps, int stokes);
        void Deallocate(void);
        void SendToDisk(int idx, header_f head, std::string outdir);
        void SendToRam(int idx, cudaStream_t &stream, int host_jump);
        BufferType **GetFilPointer(void) {return this->dfilterbank_;};
        int CheckReadyBuffer();
        void GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);
        void Update(ObsTime frame_time);
        void write(BufferType *d_data, ObsTime frame_time, unsigned int amount, cudaStream_t stream);
        // add deleted copy, move, etc constructors
};

template<class BufferType>
Buffer<BufferType>::Buffer(int id) : gpuid_(id)
{
    cudaSetDevice(gpuid_);
    start_ = 0;
    end_ = 0;
}

template<class BufferType>
Buffer<BufferType>::Buffer(int nogulps_u, size_t extrasamples_u, size_t gulpsamples_u, size_t size_u, int id) : extrasamples_(extrasamples_u),
                                                                                gulpsamples_(gulpsamples_u),
                                                                                nogulps_(nogulps_u),
                                                                                totalsamples_(size_u),
                                                                                gpuid_(id)
{
    start_ = 0;
    end_ = 0;
    state_ = new unsigned int[(int)totalsamples_];
    std::fill(state_, state_ + totalsamples_, 0);
}

template<class BufferType>
Buffer<BufferType>::~Buffer()
{
    end_ = 0;
}

template<class BufferType>
void Buffer<BufferType>::Allocate(int accumulate, size_t extra, size_t gulp, int filchans, int gulps, int stokes)
{
    fil_saved_ = 0;
    accumulate_ = accumulate;
    extrasamples_ = extra;
    gulpsamples_ = gulp;
    nochans_ = filchans;
    nogulps_ = gulps;
    nostokes_ = stokes;
    // size for a single Stokes parameter
    totalsamples_ = nogulps_ * gulpsamples_ + extrasamples_;

    std::cout << totalsamples_ << std::endl;
    std::cout << totalsamples_ * nochans_ << std::endl;

    gulptimes_ = new ObsTime[nogulps_];
    hdfilterbank_ = new BufferType*[nostokes_];
    state_ = new unsigned int[(int)totalsamples_];
    cudaCheckError(cudaHostAlloc((void**)&rambuffer_, nostokes_ * sizeof(BufferType*), cudaHostAllocDefault));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&hdfilterbank_[istoke], totalsamples_ * nochans_ * sizeof(BufferType)));
        cudaCheckError(cudaHostAlloc((void**)&rambuffer_[istoke], 2 * (gulpsamples_ + extrasamples_) * nochans_ * sizeof(BufferType), cudaHostAllocDefault));
        std::cout << "Stokes " << istoke << " done" << std::endl;
    }

    cudaCheckError(cudaMalloc((void**)&dfilterbank_, nostokes_ * sizeof(BufferType*)));
    cudaCheckError(cudaMemcpy(dfilterbank_, hdfilterbank_, nostokes_ * sizeof(BufferType*), cudaMemcpyHostToDevice));
    std::cout << "Other memory done" << std::endl;
}

template<class T>
void Buffer<T>::Deallocate(void)
{

    cudaCheckError(cudaFree(dfilterbank_));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFreeHost(rambuffer_[istoke]));
        cudaCheckError(cudaFree(hdfilterbank_[istoke]));
    }

    cudaCheckError(cudaFreeHost(rambuffer_));

    delete [] hdfilterbank_;
    delete [] gulptimes_;
}

template<class T>
void Buffer<T>::SendToDisk(int idx, header_f header, std::string outdir)
{
    SaveFilterbank(rambuffer_, gulpsamples_ + extrasamples_, (gulpsamples_ + extrasamples_) * nochans_ * idx, header, nostokes_, fil_saved_, outdir);
    fil_saved_++;
    // need info from the telescope
}

template<class T>
int Buffer<T>::CheckReadyBuffer()
{
    std::lock_guard<mutex> addguard(statemutex_);
    // for now check only the last position for the gulp
    for (int igulp = 0; igulp < nogulps_; igulp++) {
        if (state_[(igulp + 1) * gulpsamples_ + extrasamples_ - 1] == 1)
            return (igulp + 1);
    }
    return 0;
}

/*template<class T>
void Buffer<T>::GetScaling(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs)
{
    float *d_transpose;
    cudaMalloc((void**)&d_transpose, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(float));
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        transpose<<<1,nochans_,0,stream>>>(hdfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_, d_transpose, nochans_, gulpsamples_ + extrasamples_);
        scale_factors<<<1,nochans_,0,stream>>>(d_transpose, d_means, d_rstdevs, nochans_, gulpsamples_ + extrasamples_, istoke);
    }
    cudaFree(d_transpose);
    // need this so I don't save this buffer
    statemutex_.lock();
    state_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();
}*/


template<class T>
void Buffer<T>::SendToRam(int idx, cudaStream_t &stream, int host_jump)
{
    // which half of the RAM buffer we are saving into
    host_jump *= (gulpsamples_ + extrasamples_) * nochans_;
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMemcpyAsync(rambuffer_[istoke] + host_jump, hdfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
        std::cout << "Sent stokes " << istoke << std::endl;
    }
    cudaStreamSynchronize(stream);
    std::cout << "Sent to RAM... " << std::endl;
    // dump to the host memory only - not interested in the dedisperion in the dump mode
/*
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump, pd_filterbank_[0] + (idx - 1) * gulpsamples_ * nochans_, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump + 1 * (gulpsamples_ + extrasamples_) * nochans_, pd_filterbank_[1] + (idx - 1) * gulpsamples_ * nochans_, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump + 2 * (gulpsamples_ + extrasamples_) * nochans_, pd_filterbank_[2] + (idx - 1) * gulpsamples_ * nochans_, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump + 3 * (gulpsamples_ + extrasamples_) * nochans_, pd_filterbank_[3] + (idx - 1) * gulpsamples_ * nochans_, (gulpsamples_ + extrasamples_) * nochans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
*/
    statemutex_.lock();
    // HACK: the call below is wrong - restarts the whole sample state
    //std::fill(sample_state, sample_state + totsize, 0);
    state_[idx * gulpsamples_ + extrasamples_ - 1] = 0;
    statemutex_.unlock();
}


template<class T>
void Buffer<T>::Update(ObsTime frametime)
{
    std::lock_guard<mutex> addguard(statemutex_);
    int framet = frametime.framet;
    int index = 0;
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
    for (int ii = 0; ii < accumulate_; ii++) {
        index = framet % (nogulps_ * gulpsamples_);
        state_[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        //std::cout.flush();
        // second condition is to avoid sending the second buffer when the very fisrt buffer is being filled
        if ((index < extrasamples_) && (framet > extrasamples_)) {
            state_[index + nogulps_ * gulpsamples_] = 1;
        }
        framet++;
    }
}
#endif
