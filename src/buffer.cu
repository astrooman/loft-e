#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "buffer.cuh"
#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"
#include "obs_time.hpp"

Buffer::Buffer(int id) : gpuid_(id) {
    cudaSetDevice(gpuid_);
    start_ = 0;
    end_ = 0;
}


Buffer::Buffer(int nogulps_u, size_t extrasamples_u, size_t gulpsamples_u, size_t size_u, int id) : extrasamples_(extrasamples_u),
                                                                                gulpsamples_(gulpsamples_u),
                                                                                nogulps_(nogulps_u),
                                                                                totalsamples_(size_u),
                                                                                gpuid_(id) {
    start_ = 0;
    end_ = 0;
    state_ = new unsigned int[(int)totalsamples_];
    std::fill(state_, state_ + totalsamples_, 0);
}

Buffer::~Buffer() {
    end_ = 0;
}

void Buffer::Allocate(int accumulate, size_t extra, size_t gulp, int filchans, int gulps, int stokes, int perframe, int filbits) {
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
    typebytes_ = filbits / 8;

    std::cout << totalsamples_ << std::endl;
    std::cout << totalsamples_ * nochans_ << std::endl;

    gulptimes_ = new ObsTime[nogulps_];
    hdfilterbank_ = new unsigned char*[nostokes_];
    state_ = new unsigned int[(int)totalsamples_];
    cudaCheckError(cudaHostAlloc((void**)&rambuffer_, nostokes_ * sizeof(unsigned char*), cudaHostAllocDefault));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&hdfilterbank_[istoke], totalsamples_ * nochans_ * typebytes_));
        cudaCheckError(cudaHostAlloc((void**)&rambuffer_[istoke], 2 * (gulpsamples_ + extrasamples_) * nochans_ * typebytes_, cudaHostAllocDefault));
        std::cout << "Stokes " << istoke << " done" << std::endl;
    }

    cudaCheckError(cudaMalloc((void**)&dfilterbank_, nostokes_ * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dfilterbank_, hdfilterbank_, nostokes_ * sizeof(unsigned char*), cudaMemcpyHostToDevice));
    std::cout << "Other memory done" << std::endl;
}

void Buffer::Deallocate(void) {

    cudaCheckError(cudaFree(dfilterbank_));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFreeHost(rambuffer_[istoke]));
        cudaCheckError(cudaFree(hdfilterbank_[istoke]));
    }

    cudaCheckError(cudaFreeHost(rambuffer_));

    delete [] hdfilterbank_;
    delete [] gulptimes_;
}

int Buffer::CheckReadyBuffer(void) {
    std::lock_guard<mutex> addguard(statemutex_);
    // for now check only the last position for the gulp
    for (int igulp = 0; igulp < nogulps_; igulp++) {
        if (state_[(igulp + 1) * gulpsamples_ + extrasamples_ - 1] == 1)
            return (igulp + 1);
    }
    return 0;
}

void Buffer::SendToDisk(int idx, header_f header, std::string telescope, std::string outdir) {
    SaveFilterbank(rambuffer_, gulpsamples_ + extrasamples_, (gulpsamples_ + extrasamples_) * nochans_ * typebytes_ * idx, header, nostokes_, fil_saved_, telescope, outdir);
    fil_saved_++;
}

void Buffer::SendToRam(int idx, cudaStream_t &stream, int host_jump) {
    // which half of the RAM buffer we are saving into
    host_jump *= (gulpsamples_ + extrasamples_) * nochans_ * typebytes_;
    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMemcpyAsync(rambuffer_[istoke] + host_jump, hdfilterbank_[istoke] + (idx - 1) * gulpsamples_ * nochans_ * typebytes_, (gulpsamples_ + extrasamples_) * nochans_ * typebytes_, cudaMemcpyDeviceToHost, stream));
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

void Buffer::Update(ObsTime frametime) {
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
        if ((index < extrasamples_) && (filtime > extrasamples_)) {
            state_[index + nogulps_ * gulpsamples_] = 1;
        }
        filtime++;
    }
}

ObsTime Buffer::GetTime(int idx) {
    return gulptimes_[idx];
}
