#ifndef _H_PAFRB_POOL_MULTI
#define _H_PAFRB_POOL_MULTI

/*! \file pool_multi.cuh
    \brief Defines classes that are responsible for all the work done

*/

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/DedispPlan.hpp"

using boost::asio::ip::udp;
using std::mutex;
using std::pair;
using std::queue;
using std::thread;
using std::vector;

class GPUpool;

/*! \class Oberpool
    \brief Main pool class, containter for GPUpool(s).

*/

class Oberpool
{
    private:

        int ngpus;

        std::vector<std::unique_ptr<GPUpool>> gpuvector;
        std::vector<std::thread> threadvector;
    protected:

    public:
        Oberpool(void) = delete;
        Oberpool(config_s config);
        Oberpool(const Oberpool &inpool) = delete;
        Oberpool& operator=(const Oberpool &inpool) = delete;
        Oberpool(Oberpool &&inpool) = delete;
        Oberpool& operator=(Oberpool &&inpool) = delete;
        ~Oberpool(void);
        static void signal_handler(int signum);
};

// TODO: clean this mess up!!

/*! \class GPUpool
    \brief Class responsible for managing the work on a single GPU.

*/

class GPUpool
{
    private:
        // that can be anything, depending on how many output bits we decide to use
        //Buffer<float> mainbuffer;
        std::unique_ptr<Buffer<float>> p_mainbuffer;
        //DedispPlan dedisp;
        std::unique_ptr<DedispPlan> p_dedisp;
        hd_pipeline pipeline;
        hd_params params;
        int packcount;
        vector<thrust::device_vector<float>> dv_time_scrunch;   //!< Time scrunched buffer, addtime() kernel output, addchannel() kernel input; holds GPUpool::nostreams device vectors, each holding GPUpool::d_time_scrunch_size * GPUpool::stokes elements
        vector<thrust::device_vector<float>> dv_freq_scrunch;   //!< Frequency scrunched buffer, addchannel() kernel output; holds GPUpool::nostreams device vectors, each holding GPUpool::d_freq_scrunch_size * GPUpool::stokes elements

        // networking
        unsigned char **rec_bufs;
        int *sfds;

/*
        float *pdv_power;
        float *pdv_time_scrunch;
        float *pdv_freq_scrunch;
*/
        // TEXTURING
        cudaChannelFormatDesc cdesc;
        cudaArray **d_array2Dp;
        cudaResourceDesc *rdesc;
        cudaTextureDesc *tdesc;
        cudaTextureObject_t *texObj;

        // MUTEXES
        mutex buffermutex;
        mutex printmutex;
        mutex workermutex;

        // SCALING
        bool scaled_;
        float **h_means_;
        float **h_stdevs_;
        float **d_means_;
        float **d_rstdevs_;

        obs_time start_time;
        config_s config_;
        bool *bufidx_array;
        bool working;
        bool verbose_;
        static bool working_;
        bool buffer_ready[2];
        bool worker_ready[2];
        int worker_frame[2];
        // const to be safe
        // keep d_in_size and d_fft_size separate just in case the way the fft is done changes
        const unsigned int d_rearrange_size;
        const unsigned int d_in_size;               //!< size of single fft * # 1MHz channels * # time samples to average * # polarisations
        const unsigned int d_fft_size;              //!< size of single fft * # 1MHz channels * # time samples to average * # polarisations        const unsigned int d_time_scrunch_size;     //!< (size of single fft - 5) * # 1MHz channels
        const unsigned int d_time_scrunch_size;      //!< (size of single fft - 5) * # 1MHz channels
        const unsigned int d_freq_scrunch_size;     //!< d_time_scrunch_size / # frequency channels to average
        const unsigned int accumulate;              //!< The number of 108us chunks to accumulate for the GPU processing
        const unsigned int batchsize;               //!< The number of 1MHz channels to process
        const unsigned int fftpoint;                //!< Size of the single FFT
        const unsigned int filchans_;                //!< Number of output filterbank channels
        const unsigned int nostreams;               //!< Number of CUDA streams to use
        const unsigned int timeavg;                 //!< Number of post-FFT time samples to average
        const unsigned int freqavg;                 //!< Number of post-FFT frequency channels to average
        const unsigned int npol;                    //!< Number of polarisations in the input data
        unsigned int nchans;                        //!< Number of 1MHz channels in the input data
        unsigned int stokes;                        //!< Number of stoke parameters to generate and store
        // one buffer
        unsigned int filsize;
        // polarisations buffer
        unsigned char *h_pol;           //!< Buffer for semi-arranged packets for the whole bandwidth and 128 time samples
        int gpuid_;                      //!< Self-explanatory
        int pol_begin;
        int *frame_times;               //!< Array for the absolute frame numbers for given buffers
        // GPU and thread stuff
        // raw voltage buffers
        // d_in is a cufftExecC2C() input
        unsigned char *h_in = 0;        //!< Raw voltage host buffer, async copied to d_in in the GPUpool::worker()
        cufftComplex *d_in = 0;         //!< Raw voltage device buffer, cufftExecC2C() input, holds GPUpool::d_in_size * GPUpool::nostreams elements
        // the ffted signal buffer
        // cufftExecC2C() output
        // powerscale() kernel input
        cufftComplex *d_fft;            //!< Buffer for the signal after the FFT, powerscale() kernel input, holds GPUpool::d_fft_size * GPUpool::nostreams elements
        // the detected signal buffer
        // powerscale() kernel output
        // addtime() kernel input
        float *d_power;                 //!< No longer in use
        // the time scrunched signal buffer
        // addtime() kernel output
        // addchannel() kernel input
        float *d_time_scrunch;          //!< No longer in use
        // the frequency schruned signal buffer
        // addchannel() kernel output
        float *d_freq_scrunch;          //!< No longer in use
        unsigned char *d_dedisp;        //!< Not in use in the dump mode
        unsigned char *d_search;        //!< Not in use in the dump mode

        unsigned int gulps_sent;
        unsigned int gulps_processed;
        int sizes[1];                   //<! Used to store GPUpool::fftpoint - cufftPlanMany() requirement
        int avt;
        int beamno;
        int poolid_;
        size_t highest_buf;
        size_t highest_frame;
        cudaStream_t *mystreams;        //<! Pointer to the array of CUDA streams
        cufftHandle *myplans;           //<! Pointer to the array of cuFFT plans
        mutex datamutex;
        mutex workmutex;
        unsigned int *CUDAthreads;      //<! Pointer to the array of thread layouts for different kernels
        unsigned int *CUDAblocks;       //<! Pointer to the array of block layouts for different kernels
        // containers
        // use queue as FIFO needed
        queue<pair<vector<cufftComplex>, obs_time>> mydata;
        //queue<vector<cufftComplex>> mydata;
        vector<thread> mythreads;
        vector<thread> receive_threads;
        unsigned int dedisp_buffno;
        int pack_per_buf;
        size_t dedisp_buffsize;
        size_t dedisp_totsamples;
        std::string strip;
    protected:

    public:
        //! A default constructor.
        /*!
            Deleted
        */
        GPUpool(void) = delete;
        //! A constructor.
        /*!
            \param id the GPU id to be set using cudaSetDevice()
            \param config the configuration structure
        */
        GPUpool(int id, config_s config);
        ~GPUpool(void);
        //! A copy constructor.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GPUpool(const GPUpool &inpool) = delete;
        //! An assignment operator.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GPUpool& operator=(const GPUpool &inpool) = delete;
        //! Move constructor.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GPUpool(GPUpool &&inpool) = delete;
        //! Move assignment operator.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GPUpool& operator=(GPUpool &&inpool) = delete;

        //! Add the data to the processing queue.
        /*! Called in GPupool::get_data() method.
            Adds a pair to the queue consistinf of the data buffer and associated time structure.
        */
        void add_data(cufftComplex *buffer, obs_time frame_time);
        //! Dedispersion thread worker
        /*! Responsible for picking up the data buffer when ready and dispatching it to the dedispersion (Buffer::send() method).
            In the filterbank dump mode, responsible for initialising the dump (Buffer::dump() method).
            \param dstream stream number, used to access stream from mystreams array
        */
        void dedisp_thread(int dstream);
        //! Main GPUpool method.
        /*! Responsible for setting up the GPU execution.
            All memory allocated here, streams, cuFFT plans threads created here as well.
        */
        void execute(void);
        //! Reads the data from the UDP packet.
        /*!
            \param *data buffer read by async_receive_from()
            \param fpga_id the FPGA number obtained from the sender's IP address; used to identify the frequency chunk and place in the buffer it will be saven in
            \param start_time structure containing the information when the current observation started (reference epoch and seconds from the reference epoch)
        */
        void get_data(unsigned char* data, int fpga_id, obs_time start_time, header_s head);
        //! Handles the SIGINT signal. Must be static.
        /*!
            \param signum signal number - should be 2 for SIGINT
        */
        static void HandleSignal(int signum);
        //! Thread responsible for running the FFT.
        /*! There are 4 such threads per GPU - 4 streams per GPU used.
            Each thread is responsible for picking up the data from the queue (the thread yields if the is no data available), running the FFT and power, time scrunch and frequency scrunch kernels.
            After successfull kernel execution, writes to the main data buffer using write() Buffer method.
        */
        void worker(int stream);
        //! Handler called from async_receive_from().
        /*! This function is responsible for handling the asynchronous receive on the socket.
            \param error error code
            \param bytes_transferred number of bytes received
            \param endpoint udp::endpoint object containing sender information (used to obtain the fpga_id from the sender's IP)

        */
        void receive_handler(const boost::system::error_code& error, std::size_t bytes_transferred, udp::endpoint endpoint);
        //! Calls async_receive_from() on the UDP socket.
        void receive_thread(int ii);
        //! Single pulse search thread worker.
        /*! Responsible for picking up the dedispersed data buffer and dispatching it to the single pulse search pipeline.
            Calls hd_execute() and saves the filterbank if the appropriate single pulse has been detected.
            Disabled in the fulterbank dump mode.
        */
        void search_thread(int sstream);
};

#endif
