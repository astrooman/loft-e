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
#include "obs_time.hpp"

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
        Oberpool(InConfig config);
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

        std::vector<int> ports_;
        std::vector<std::string> strip_;
        std::vector<std::thread> receivethreads_;

        bool verbose_;
        static bool working_;

        InConfig config_;

        int *sockfiledesc_;

        ObsTime starttime_;

        // unpacking into float for FFT purposes
        float **dunpacked_;
        float **hdunpacked_;

        unsigned char **dinpol_;
        unsigned char **hdinpol_;
        unsigned char **inpol_;
        unsigned char **recbufs_;

        unsigned int accumulate_;
        unsigned int availthreads_;
        unsigned int gpuid_;
        unsigned int inbits;
        unsigned int inpolsize_;
        unsigned int nopols_;
        unsigned int noports_;
        unsigned int nostreams_;
        unsigned int packperbuf_;
        unsigned int poolid_;
        unsigned int unpackedsize_;

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
        GPUpool(int id, InConfig config);
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
        void add_data(cufftComplex *buffer, ObsTime frame_time);
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
        void get_data(unsigned char* data, int fpga_id, ObsTime start_time, header_s head);
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
