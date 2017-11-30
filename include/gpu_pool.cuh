#ifndef _H_LOFTE_GPU_POOL
#define _H_LOFTE_GPU_POOL

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/DedispPlan.hpp"
#include "obs_time.hpp"

// NOTE: DADA headers
#include "dada_client.h"

struct DadaContext {
    bool verbose;
    bool headerwritten;
    dada_hdu_t *dhdu;
    multilog_t *mlog;
    cudaStream_t stream;
    unsigned char *devicememory;

    char *headerfile;
    char *obsheader;
};

class GpuPool
{
    private:

        // NOTE: We will scale everything down to 8 bits for now
        std::unique_ptr<Buffer<unsigned char>> filbuffer_;
        std::unique_ptr<DedispPlan> dedispplan_;

        std::vector<int> ports_;
        std::vector<std::string> strip_;
        std::vector<std::thread> gputhreads_;
        std::vector<std::thread> receivethreads_;

        bool verbose_;
        static bool working_;
        bool scaled_;

        double freqtop_;
        double freqoff_;
        double samptime_;

        InConfig config_;

        ObsTime starttime_;

        thrust::device_vector<float> dmeans_;
        thrust::device_vector<float> dstdevs_;
        thrust::device_vector<float> dfactors_;

        unsigned int availthreads_;
        unsigned int avgfreq_;
        unsigned int dedispextrasamples_;
        unsigned int dedispgulpsamples_;
        unsigned int fftbatchsize_;
        unsigned int fftpoints_;
        unsigned int fftsize_;
        unsigned int gpuid_;
        unsigned int rawbuffersize_;
        unsigned int rawgpubuffersize_;
        unsigned int nogulps_;
        unsigned int nopols_;
        unsigned int noports_;
        unsigned int nostokes_;
        unsigned int nostreams_;
        unsigned int packperbuf_;
        unsigned int perblock_;
        unsigned int poolid_;
        unsigned int sampperthread_;
        unsigned int scalesamples_;
        unsigned int unpackedsize_;

        bool *readyrawidx_;

        cudaStream_t dedispstream_;
        cudaStream_t *gpustreams_;

        cufftComplex **dfft_;
        cufftComplex **hdfft_;
        cufftHandle *fftplans_;

        // unpacking into float for FFT purposes
        float **hdmeans_;
        float **hdstdevs_;
        float **dunpacked_;
        float **hdunpacked_;
        float *pdfactors_;
        float *pdmeans_;
        float *pdstdevs_;

        int *fftsizes_;
        int *sockfiledesc_;

        std::string telescope_;

        unsigned char **dinpol_;
        unsigned char **hdinpol_;
        unsigned char **inpol_;
        unsigned char **recbufs_;

        unsigned int *cudablocks_;
        unsigned int *cudathreads_;
        unsigned int *framenumbers_;

    protected:

    public:
        //! A default constructor.
        /*!
            Deleted
        */
        GpuPool(void) = delete;
        //! A constructor.
        /*!
            \param id the GPU id to be set using cudaSetDevice()
            \param config the configuration structure
        */
        GpuPool(int id, InConfig config);
        ~GpuPool(void);
        //! A copy constructor.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GpuPool(const GpuPool &inpool) = delete;
        //! An assignment operator.
        /*!
            Deleted for safety, to avoid problems with shallow copying.
        */
        GpuPool& operator=(const GpuPool &inpool) = delete;
        //! Move constructor.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GpuPool(GpuPool &&inpool) = delete;
        //! Move assignment operator.
        /*!
            Deleted. Can't really be bothered with moving at this stage.
        */
        GpuPool& operator=(GpuPool &&inpool) = delete;

        //! Add the data to the processing queue.
        /*! Called in GPupool::get_data() method.
            Adds a pair to the queue consistinf of the data buffer and associated time structure.
        */
        void add_data(cufftComplex *buffer, ObsTime frame_time);
        //! Dedispersion thread worker
        /*! Responsible for picking up the data buffer when ready and dispatching it to the dedispersion (Buffer::send() method).
            In the filterbank dump mode, responsible for initialising the dump (Buffer::dump() method).
            \param dstream stream number, used to access stream from mystreams array
        */
        void SendForDedispersion(cudaStream_t dstream);
        //! Main GpuPool method.
        /*! Responsible for setting up the GPU execution.
            All memory allocated here, streams, cuFFT plans threads created here as well.
        */
        void Initialise(void);
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
        void DoGpuWork(int stream);
        //! Handler called from async_receive_from().
        /*! This function is responsible for handling the asynchronous receive on the socket.
            \param error error code
            \param bytes_transferred number of bytes received
            \param endpoint udp::endpoint object containing sender information (used to obtain the fpga_id from the sender's IP)

        */
        //! Calls async_receive_from() on the UDP socket.
        void ReceiveData(int portid, int recport);
        //! Single pulse search thread worker.
        /*! Responsible for picking up the dedispersed data buffer and dispatching it to the single pulse search pipeline.
            Calls hd_execute() and saves the filterbank if the appropriate single pulse has been detected.
            Disabled in the fulterbank dump mode.
        */
        void search_thread(int sstream);
};

#endif
