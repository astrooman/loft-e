#ifndef _H_PAFRB_POOL
#define _H_PAFRB_POOL

/*! \file pool.cuh
    \brief No longer in use - call Oberpool with -n 1 option for a singlpe-GPU execution.
*/


#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <buffer.hpp>
#include <config.hpp>
#include <cuda.h>
#include <cufft.h>
#include <dedisp/DedispPlan.hpp>

#include <thrust/device_vector.h>

using std::mutex;
using std::pair;
using std::queue;
using std::thread;
using std::vector;

class Pool
{
    private:
        // that can be anything, depending on how many output bits we decide to use
        Buffer<float> mainbuffer;
        DedispPlan dedisp;
        hd_pipeline pipeline;
        hd_params params;

        vector<thrust::device_vector<float>> dv_power;
        vector<thrust::device_vector<float>> dv_time_scrunch;
        vector<thrust::device_vector<float>> dv_freq_scrunch;

/*
        float *pdv_power;
        float *pdv_time_scrunch;
        float *pdv_freq_scrunch;
*/

        bool working;
        // const to be safe
        // keep d_in_size and d_fft_size separate just in case the way the fft is done changes
        const unsigned int d_in_size;                // size of single fft * # 1MHz channels * # time samples to average * # polarisations
        const unsigned int d_fft_size;              // size of single fft * # 1MHz channels * # time samples to average * # polarisations
        const unsigned int d_power_size;            // d_fft_size / # polarisations
        const unsigned int d_time_scrunch_size;     // (size of single fft - 5) * # 1MHz channels
        const unsigned int d_freq_scrunch_size;     // d_time_scrunch_size / # frequency channels to average
        const unsigned int batchsize;               // the number of FFTs to process at one
        const unsigned int fftpoint;                // size of the single FFT
        const unsigned int nostreams;               // # CUDA streams
        const unsigned int timeavg;                 // # time samples to average
        const unsigned int freqavg;                 // # frequency channels to average
        const unsigned int npol;                    // number of polarisations in the input data
        unsigned int nchans;
        unsigned int stokes;
        // one buffer
        unsigned int filsize;
        // polarisations buffer
        cufftComplex *h_pol;
        int pol_begin;
        // GPU and thread stuff
        // raw voltage buffers
        // d_in is a cufftExecC2C() input
        cufftComplex *h_in, *d_in;
        // the ffted signal buffer
        // cufftExecC2C() output
        // powerscale() kernel input
        cufftComplex *d_fft;
        // the detected signal buffer
        // powerscale() kernel output
        // addtime() kernel input
        float *d_power;
        // the time scrunched signal buffer
        // addtime() kernel output
        // addchannel() kernel input
        float *d_time_scrunch;
        // the frequency schruned signal buffer
        // addchannel() kernel output
        float *d_freq_scrunch;          // the frequency scrunched signal buffer
        unsigned char *d_dedisp;        // dedispersion buffer - aggregated frequency scrunched buffer
        unsigned char *d_search;        // single pulse search buffer - dedispersion output

        unsigned int gulps_sent;
        unsigned int gulps_processed;
        int sizes[1];
        int avt;
        cudaStream_t *mystreams;
        cufftHandle *myplans;
        mutex datamutex;
        mutex workmutex;
        unsigned int *nthreads;
        unsigned int *nblocks;
        // containers
        // use queue as FIFO needed
        queue<pair<vector<cufftComplex>, obs_time>> mydata;
        //queue<vector<cufftComplex>> mydata;
        vector<thread> mythreads;
        unsigned int dedisp_buffno;
        int pack_per_buf;
        size_t dedisp_buffsize;
        size_t dedisp_totsamples;
    protected:

    public:
        Pool(unsigned int bs, unsigned int fs, unsigned int ts, unsigned int fr, unsigned int sn, unsigned int np, config_s config);
        ~Pool(void);
        // just in case I don't catch where they are used - not deep copy currently implemented
        Pool(const Pool &inpool) = delete;
        Pool &operator=(const Pool &inpool) = delete;

        void add_data(cufftComplex *buffer, obs_time frame_time);
        void dedisp_thread(int dstream);
        void get_data(unsigned char* data, int frame, int &highest_frame, int &highest_framet, int &thread, obs_time start_time);
        void minion(int stream);
        void receive_thread(int stream);
        void search_thread(int sstream);
};

#endif
