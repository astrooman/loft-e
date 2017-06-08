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

        std::unique_ptr<Buffer> filbuffer_;
        std::unique_ptr<DedispPlan> dedispplan_;

        std::vector<int> ports_;
        std::vector<std::string> strip_;
        std::vector<std::thread> gputhreads_;
        std::vector<std::thread> receivethreads_;

        bool verbose_;
        static bool working_;
        bool scaled_;

        const unsigned int filbits_;
        const unsigned int headlen_;
        const unsigned int vdiflen_;

        double freqtop_;
        double freqoff_;
        double samptime_;

        InConfig config_;

        ObsTime starttime_;

        unsigned int accumulate_;
        unsigned int availthreads_;
        unsigned int avgfreq_;
        unsigned int avgtime_;
        unsigned int dedispextrasamples_;
        unsigned int dedispgulpsamples_;
        unsigned int fftbatchsize_;
        unsigned int fftpoints_;
        unsigned int fftsize_;
        unsigned int filchans_;
        unsigned int gpuid_;
        unsigned int inbits_;
        unsigned int inpolbufsize_;
        unsigned int inpolgpusize_;
        unsigned int nogulps_;
        unsigned int nopols_;
        unsigned int noports_;
        unsigned int nostokes_;
        unsigned int nostreams_;
        unsigned int packperbuf_;
        unsigned int perblock_;
        unsigned int poolid_;
        unsigned int powersize_;
        unsigned int rem_;
        unsigned int sampperthread_;
        unsigned int scaledsize_;
        unsigned int unpackedsize_;

        bool *readybufidx_;

        cudaStream_t dedispstream_;
        cudaStream_t *gpustreams_;

        cufftComplex **dfft_;
        cufftComplex **hdfft_;
        cufftHandle *fftplans_;

        // unpacking into float for FFT purposes
        float **dmeans_;
        float **dstdevs_;
        float **hdmeans_;
        float **hdstdevs_;
        float **dunpacked_;
        float **dpower_;
        float **hdunpacked_;
        float **hdpower_;

        int *fftsizes_;
        int *sockfiledesc_;

        std::string telescope_;

        unsigned char **dinpol_;
        unsigned char **hdinpol_;
        unsigned char **inpol_;
        unsigned char **recbufs_;
        // TODO: this should really be template-like - we may choose to scale to different number of bits
        unsigned char **dscaled_;
        unsigned char **hdscaled_;

        unsigned int *cudablocks_;
        unsigned int *cudathreads_;
        unsigned int *framenumbers_;

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
        void SendForDedispersion(cudaStream_t dstream);
        //! Main GPUpool method.
        /*! Responsible for setting up the GPU execution.
            All memory allocated here, streams, cuFFT plans threads created here as well.
        */
        void Initialise(void);
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
        template <class FilType>
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

template<class OutType>
void GPUpool::DoGpuWork(int stream)
{
    // let us hope one stream will be enough or we will have to squeeze multiple streams into single CPU core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)gpuid_ * 3 + 1, &cpuset);

    int retaff = pthread_setaffinity_np(gputhreads_[stream].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        std::cerr << "Error setting thread affinity for the GPU processing, stream " << stream << std::endl;
    }

    if (verbose_) {
        std::cout << "Starting worker " << gpuid_ << ":" << stream << " on CPU " << sched_getcpu() << std::endl;
    }

    cudaCheckError(cudaSetDevice(gpuid_));

    bool innext{false};
    bool inthis{false};

    int bufferidx;
    int checkidx{5};    // hav many receive buffer samples to probe for the existance of the packet
    int startnext;
    int startthis;

    ObsTime frametime;

    unsigned int perframe = vdiflen_ * 8 / inbits_ / (2 * fftpoints_) / avgtime_;     // the number of output time samples per frame

    int donebuffers = 0;

    while (working_) {
        // TODO: there must be a way to simplify this
        // I should not be allowed to write code
	// startthis check the end of one buffer
	// startnext checks the beginning of the buffer after it
        if (nostreams_ == 1) {      // different behaviour if we are using one stream only - need to check both buffers
            for (int ibuffer = 0; ibuffer < 2; ibuffer++) {
                bufferidx = ibuffer;
                startthis = (ibuffer + 1) * accumulate_ - checkidx;
                startnext = ((ibuffer + 1) % 2) * accumulate_; // + checkidx;

                inthis = inthis || readybufidx_[startthis + checkidx - 1];
                innext = innext || readybufidx_[startnext + checkidx];

                /*for (int iidx = 0; iidx < checkidx; iidx++) {
                    inthis = inthis && readybufidx_[startthis + iidx];
                    innext = innext && readybufidx_[startnext + iidx];
                } */

		if (inthis && innext)
			break;

		inthis = false;
		innext = false;
            }
        } else {
            bufferidx = stream;
            startthis = (stream + 1) * accumulate_ - checkidx;
            startnext = ((stream + 1) % nostreams_) * accumulate_; // + checkidx;

            for (int iidx = 0; iidx < checkidx; iidx++) {
                inthis = inthis || readybufidx_[startthis + iidx];
                innext = innext || readybufidx_[startnext + iidx];
            }
        }

        if (inthis && innext) {

            readybufidx_[startthis + checkidx - 1] = false;
            readybufidx_[startnext + checkidx] = false;

            /* for (int iidx = 0; iidx < checkidx; iidx++) {
                // need to restart the state of what we have already sent for processing
                readybufidx_[startthis + iidx] = false;
                readybufidx_[startnext + iidx] = false;
            } */

            //cout << "Have the buffer " << bufferidx << ":" << startthis << ":" << startnext << endl;
            //cout.flush();
            innext = false;
            inthis = false;
            frametime.startepoch = starttime_.startepoch;
            frametime.startsecond = starttime_.startsecond;
            frametime.framet = framenumbers_[bufferidx / nostreams_ * accumulate_];
            //cout << frametime.framet << endl;
            for (int ipol = 0; ipol < nopols_; ipol++) {
                cudaCheckError(cudaMemcpyAsync(hdinpol_[ipol] + stream * inpolgpusize_ / nostreams_, inpol_[ipol] + bufferidx * inpolgpusize_ / nostreams_, accumulate_ * vdiflen_ * sizeof(unsigned char), cudaMemcpyHostToDevice, gpustreams_[stream]));
            }

            UnpackKernel<<<cudablocks_[0], cudathreads_[0], 0, gpustreams_[stream]>>>(dinpol_, dunpacked_, nopols_, sampperthread_, rem_, accumulate_ * vdiflen_, 8 / inbits_);
/*            std::ofstream unpackedfile("unpacked.dat");

            float *outunpacked = new float[unpackedsize_];
            cudaCheckError(cudaMemcpy(outunpacked, hdunpacked_[0], unpackedsize_ * sizeof(float), cudaMemcpyDeviceToHost));

            for (int isamp = 0; isamp < unpackedsize_; isamp++) {
                unpackedfile << outunpacked[isamp] << endl;
            }

            delete [] outunpacked;
            unpackedfile.close();
*/
            for (int ipol = 0; ipol < nopols_; ipol++) {
                cufftCheckError(cufftExecR2C(fftplans_[stream], hdunpacked_[ipol] + stream * unpackedsize_ / nostreams_, hdfft_[ipol] + stream * fftsize_ / nostreams_));
                //cufftCheckError(cufftExecR2C(fftplans_[stream], hdunpacked_[ipol], hdfft_[ipol]));
            }
/*            std::ofstream fftedfile("ffted.dat");

            cufftComplex *outfft = new cufftComplex[fftsize_];
            cudaCheckError(cudaMemcpy(outfft, hdfft_[0], fftsize_ * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

            for (int isamp = 0; isamp < fftsize_; isamp++) {
                fftedfile << outfft[isamp].x * outfft[isamp].x + outfft[isamp].y * outfft[isamp].y << endl;
            }

            delete [] outfft;

            fftedfile.close();
            working_ = false;
*/
            //PowerScaleKernel<<<cudablocks_[1], cudathreads_[1], 0, gpustreams_[stream]>>>(dfft_, dscaled_, dmeans_, dstdevs_, avgfreq_, avgtime_, filchans_, perblock_, stream * fftsize_, stream * scaledsize_);
            // this version should really be used, as we want to save directly into the filterbank buffer
            PowerScaleKernel<OutType><<<cudablocks_[1], cudathreads_[1], 0, gpustreams_[stream]>>>(dfft_, filbuffer_ -> GetFilPointer(), dmeans_, dstdevs_, avgfreq_, avgtime_, filchans_, perblock_, stream * fftsize_ / nostreams_, nogulps_, dedispgulpsamples_, dedispextrasamples_, frametime.framet, perframe);
            filbuffer_ -> Update(frametime);
            // ScaleKernel<<<cudablocks_[2], cudathreads_[2], 0, gpustreams_[stream]>>>(dpower_, filbuffer_ -> GetFilPointer());
            cudaCheckError(cudaDeviceSynchronize());
            donebuffers++;
            //if (donebuffers > 5)
            //    working_ = false;
        } else {
            std::this_thread::yield();
        }
    }
}

#endif
