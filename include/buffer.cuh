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
        BufferType **pd_filterbank_;                                          // array of raw pointers to Stoke parameters device vectors
        BufferType **ph_filterbank_;                                          // same as above but for host vector
        BufferType *ph_fil_;
        size_t totsize_;            // total size of the data: #gulps * gulp size + extra samples for dedispersion
        size_t gulp_;            // size of the single gulp
        size_t extra_;           // number of extra time samples required to process the full gulp
        int accumulate_;
        int gpuid_;
        int gulpno_;             // number of gulps required in the buffer
        int nchans_;             // number of filterbank channels per time sample
        int stokes_;             // number of Stokes parameters to keep in the buffer
        int fil_saved_;
        mutex buffermutex_;
        mutex statemutex_;
        size_t start_;
        size_t end_;
        ObsTime *gulp_times_;
        // TODO: do we still need that?
        BufferType *d_buf_;
        unsigned int *state_;     // 0 for no data, 1 for data
    protected:

    public:
        Buffer(int id);
        Buffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id);
        ~Buffer(void);

        void allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u);
        void deallocate(void);
        void dump(int idx, header_f head, std::string outdir);
        float **get_pfil(void) {return this->pd_filterbank_;};
        int ready();
        void rescale(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs);
        void send(unsigned char *out, int idx, cudaStream_t &stream, int host_jump);
        void update(ObsTime frame_time);
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
Buffer<BufferType>::Buffer(int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int id) : extra_(extra_u),
                                                                                gulp_(gulp_u),
                                                                                gulpno_(gulpno_u),
                                                                                totsize_(size_u),
                                                                                gpuid_(id)
{
    start_ = 0;
    end_ = 0;
    state_ = new unsigned int[(int)totsize_];
    std::fill(state_, state_ + totsize_, 0);
}

template<class BufferType>
Buffer<BufferType>::~Buffer()
{
    end_ = 0;
}

template<class BufferType>
void Buffer<BufferType>::allocate(int acc_u, int gulpno_u, size_t extra_u, size_t gulp_u, size_t size_u, int filchans, int stokes_u)
{
    fil_saved_ = 0;
    accumulate_ = acc_u;
    extra_ = extra_u;
    gulp_ = gulp_u;
    gulpno_ = gulpno_u;
    nchans_ = filchans;
    // size is the size of the buffer for the single Stokes parameter
    totsize_ = size_u;
    stokes_ = stokes_u;
    gulp_times_ = new ObsTime[gulpno_];
    h_filterbank_.resize(stokes_);
    d_filterbank_.resize(stokes_);
    pd_filterbank_ = new float*[stokes_];
    ph_filterbank_ = new float*[stokes_];
    for (int ii = 0; ii < stokes_; ii++) {
        // used to hold 2 full filterbank buffers
        h_filterbank_[ii].resize((gulp_ + extra_) * 2 * nchans_);
        ph_filterbank_[ii] = thrust::raw_pointer_cast(h_filterbank_[ii].data());
        d_filterbank_[ii].resize(totsize_ * nchans_);
        pd_filterbank_[ii] = thrust::raw_pointer_cast(d_filterbank_[ii].data());
    }
    cudaCheckError(cudaMalloc((void**)&d_buf_, totsize_ * stokes_ * sizeof(BufferType)));
    state_ = new unsigned int[(int)totsize_];
    cudaCheckError(cudaHostAlloc((void**)&ph_fil_, (gulp_ + extra_) * nchans_ * stokes_ * 2 * sizeof(float), cudaHostAllocDefault));
    std::fill(state_, state_ + totsize_, 0);
}

template<class T>
void Buffer<T>::deallocate(void)
{
    cudaCheckError(cudaFreeHost(ph_fil_));
    cudaCheckError(cudaFree(d_buf_));
    delete [] state_;
    delete [] gulp_times_;
    delete [] pd_filterbank_;
    delete [] ph_filterbank_;
}

template<class T>
void Buffer<T>::dump(int idx, header_f header, std::string outdir)
{
        save_filterbank2(ph_fil_, gulp_ + extra_, (gulp_ + extra_) * nchans_ * stokes_ * idx, header, stokes_, fil_saved_, outdir);
        fil_saved_++;
        // need info from the telescope
}

template<class T>
int Buffer<T>::ready()
{
    std::lock_guard<mutex> addguard(statemutex_);
    // for now check only the last position for the gulp
    for (int ii = 0; ii < gulpno_; ii++) {
        if (state_[(ii + 1) * gulp_ + extra_ - 1] == 1)
            return (ii + 1);
    }
    return 0;
}

template<class T>
void Buffer<T>::rescale(int idx, cudaStream_t &stream, float **d_means, float **d_rstdevs)
{
    float *d_transpose;
    cudaMalloc((void**)&d_transpose, (gulp_ + extra_) * nchans_ * sizeof(float));
    for (int ii = 0; ii < stokes_; ii++) {
        transpose<<<1,nchans_,0,stream>>>(pd_filterbank_[ii] + (idx - 1) * gulp_ * nchans_, d_transpose, nchans_, gulp_ + extra_);
        scale_factors<<<1,nchans_,0,stream>>>(d_transpose, d_means, d_rstdevs, nchans_, gulp_ + extra_, ii);
    }
    cudaFree(d_transpose);
    // need this so I don't save this buffer
    statemutex_.lock();
    state_[idx * gulp_ + extra_ - 1] = 0;
    statemutex_.unlock();
}


template<class T>
void Buffer<T>::send(unsigned char *out, int idx, cudaStream_t &stream, int host_jump)
{
    // which half of the RAM buffer we are saving into
    host_jump *= (gulp_ + extra_) * nchans_ * stokes_;
    // dump to the host memory only - not interested in the dedisperion in the dump mode
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump, pd_filterbank_[0] + (idx - 1) * gulp_ * nchans_, (gulp_ + extra_) * nchans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump + 1 * (gulp_ + extra_) * nchans_, pd_filterbank_[1] + (idx - 1) * gulp_ * nchans_, (gulp_ + extra_) * nchans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump + 2 * (gulp_ + extra_) * nchans_, pd_filterbank_[2] + (idx - 1) * gulp_ * nchans_, (gulp_ + extra_) * nchans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaCheckError(cudaMemcpyAsync(ph_fil_ + host_jump + 3 * (gulp_ + extra_) * nchans_, pd_filterbank_[3] + (idx - 1) * gulp_ * nchans_, (gulp_ + extra_) * nchans_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    statemutex_.lock();
    // HACK: the call below is wrong - restarts the whole sample state
    //std::fill(sample_state, sample_state + totsize, 0);
    state_[idx * gulp_ + extra_ - 1] = 0;
    statemutex_.unlock();
}


template<class T>
void Buffer<T>::write(T *d_data, ObsTime frame_time, unsigned int amount, cudaStream_t stream)
{
    // need to make sure only one stream saves the data to the buffer
    // not really a problem anyway - only one DtD available at a time
    // we will save one data sample at a time, with fixed size
    // no need to check that there is enough space available to fit all the data before the end of the buffer
    std::lock_guard<mutex> addguard(buffermutex_);
    int index = frame_time.framet % totsize_;
    if((index % gulp_) == 0)
        gulp_times_[index / gulp_] = frame_time;
    if (end_ == totsize_)    // reached the end of the buffer
        end_ = end_ - gulpno_ * gulp_;    // go back to the start

    // TODO: try to come up with a slightly different implementation - DtD copies should be avoided whenever possible
    cudaCheckError(cudaMemcpyAsync(pd_filterbank_[0] + index * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    cudaCheckError(cudaMemcpyAsync(pd_filterbank_[1] + index * amount, d_data + amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    //cudaCheckError(cudaMemcpyAsync(pd_filterbank_[2] + index * amount, d_data + 2 * amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    //cudaCheckError(cudaMemcpyAsync(pd_filterbank_[3] + index * amount, d_data + 3 * amount, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    cudaStreamSynchronize(stream);
    state_[index] = 1;

    // need to save in two places in the buffer
    if (index >= gulpno_ * gulp_) {
        // simplify the index algebra here
        // TODO: need to be actually sorted out properly
        cudaCheckError(cudaMemcpyAsync(d_buf_ + index - (gulpno_ * gulp_) * amount, d_data, amount * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        statemutex_.lock();
        state_[index - (gulpno_ * gulp_)] = 1;
        statemutex_.unlock();
    }
    end_ = end_ + amount;
}

template<class T>
void Buffer<T>::update(ObsTime frame_time)
{
    std::lock_guard<mutex> addguard(statemutex_);
    int framet = frame_time.framet;
    int index = frame_time.framet % (gulpno_ * gulp_);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
    for (int ii = 0; ii < accumulate_; ii++) {
        index = framet % (gulpno_ * gulp_);
        state_[index] = 1;
        //std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        //std::cout.flush();
        if ((index < extra_) && (framet > extra_)) {
            state_[index + gulpno_ * gulp_] = 1;
        }
        framet++;
    }
}


/*template<class T>
void Buffer<T>::update(ObsTime frame_time)
{
    std::lock_guard<mutex> addguard(statemutex_);
    int framet = frame_time.framet;
    int index = frame_time.framet % (gulpno * gulp);
    //std::cout << framet << " " << index << std::endl;
    //std::cout.flush();
    for (int ii = 0; ii < accumulate; ii++) {
        index = framet % (gulpno * gulp);
        sample_state[index] = 1;
        std::cout << framet << " " << index << " " << framet % totsize << std::endl;
        std::cout.flush();
        if ((framet % totsize) >= (gulpno * gulp)) {
            sample_state[index + gulpno * gulp] = 1;
        }
        framet++;
    }
} */
#endif
