#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <cufft.h>
#include <cuda.h>
#include <pthread.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/dedisp.hpp"
#include "dedisp/DedispPlan.hpp"
#include "errors.hpp"
#include "filterbank.hpp"
#include "get_mjd.hpp"
#include "gpu_pool.cuh"
#include "heimdall/pipeline.hpp"
#include "kernels.cuh"
#include "obs_time.hpp"
#include "print_safe.hpp"

#include <inttypes.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

using std::cerr;
using std::cout;
using std::endl;
using std::ostringstream;
using std::pair;
using std::queue;
using std::string;
using std::thread;
using std::unique_ptr;
using std::vector;

// NOTE: These values will not change between runs, so we take them out of config and make constant
#define FILCHANS 256
#define HEADLEN 32
#define INBITS 2
#define NACCUMULATE 4000
#define NOPOLS 2
#define PERBLOCK 625
#define TIMEAVG 16
#define VDIFLEN 8000

struct FactorFunctor {
    __host__ __device__ float operator()(float val) {
        return val != 0 ? 1.0f/val : val;
    }
};

bool GpuPool::working_ = true;

GpuPool::GpuPool(InConfig config) : avgfreq_(config.freqavg),
                                            config_(config),
                                            dedispgulpsamples_(config.gulp),
                                            fftpoints_(config.fftsize),
                                            freqoff_(config.foff),
                                            gpuid_(config.gpuid),
                                            nostokes_(config.nostokes),
                                            nostreams_(config.nostreams),
                                            poolid_(config.half),
                                            ports_(config.ports),
                                            samptime_(config.tsamp),
                                            scaled_(false),
                                            strip_(config.ips),
                                            telescope_("NT"),
                                            verbose_(config.verbose)
{
    availthreads_ = thread::hardware_concurrency();
    if (availthreads_ == 0) {
        cerr << "Could not obtain the number of cores for the telescope " << telescope_ << "!\n";
        cerr << "Will set to 3!";
        // NOTE: This is true for LOFT-e machines for now - be careful as it might change in the future
        availthreads_ = 3;
    }

    if (verbose_) {
        PrintSafe("Starting the GPU pool", gpuid_, "...");
    }

    noports_ = ports_.size();

    signal(SIGINT, GpuPool::HandleSignal);
    cudaCheckError(cudaSetDevice(gpuid_));

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // * 3 as there are 3 cores available for each telescope/GPU
    CPU_SET((int)(gpuid_) * 3, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for the GPU pool", poolid_);
    }

    if(verbose_) {
        PrintSafe("GPU pool", poolid_, "running on CPU", sched_getcpu());
    }

    // STAGE: memory initialisation
    if (verbose_) {
        PrintSafe("Initialising the memory on pool", poolid_, "...");
    }

    gpustreams_ = new cudaStream_t[nostreams_];

    // NOTE: We have to use one stream for the GPU filterbank processing
    // NOTE: One stream has to use two buffers side by side to cover for time spent on copying
    rawbuffersize_ = 2 * NACCUMULATE * VDIFLEN;
    readyrawidx_ = new bool[rawbuffersize_];
    framenumbers_ = new unsigned int[2 * NACCUMULATE];
    rawgpubuffersize_ = NACCUMULATE * VDIFLEN * nostreams_;
    // NOTE: 4 unpacked samples per incoming byte
    unpackedsize_ = NACCUMULATE * VDIFLEN * nostreams_ * 4;

    fftsizes_ = new int[1];
    // NOTE: Need twice as many input samples for the R2C transform
    fftsizes_[0] = 2 * FILCHANS;
    fftbatchsize_ = unpackedsize_ / (2 * FILCHANS) / nostreams_;
    // NOTE: The output of the R2C transform is  FILCHANS + 1
    fftsize_ = fftbatchsize_ * (FILCHANS + 1) * nostreams_;

    fftplans_ = new cufftHandle[nostreams_];
    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cudaCheckError(cudaStreamCreate(&gpustreams_[igstream]));
        cufftCheckError(cufftPlanMany(&fftplans_[igstream], 1, fftsizes_, NULL, 1, FILCHANS, NULL, 1, FILCHANS, CUFFT_R2C, fftbatchsize_));
        cufftCheckError(cufftSetStream(fftplans_[igstream], gpustreams_[igstream]));
    }

    hdinpol_ = new unsigned char*[NOPOLS];
    hdunpacked_ = new float*[NOPOLS];
    hdfft_ = new cufftComplex*[NOPOLS];

    cudaCheckError(cudaHostAlloc((void**)&inpol_, NOPOLS * sizeof(unsigned char*), cudaHostAllocDefault));

    for (int ipol = 0; ipol < NOPOLS; ipol++) {
        cudaCheckError(cudaHostAlloc((void**)&inpol_[ipol], rawbuffersize_ * sizeof(unsigned char), cudaHostAllocDefault));
        cudaCheckError(cudaMalloc((void**)&hdinpol_[ipol], rawgpubuffersize_ * sizeof(unsigned char)));
        // NOTE: We are unpacking 2-bit number to floats - courtesy of cuFFT
        cudaCheckError(cudaMalloc((void**)&hdunpacked_[ipol], unpackedsize_ * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&hdfft_[ipol], fftsize_ * sizeof(cufftComplex)));
    }

    cudaCheckError(cudaMalloc((void**)&dinpol_, NOPOLS * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dinpol_, hdinpol_, NOPOLS * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void**)&dunpacked_, NOPOLS * sizeof(float*)));
    cudaCheckError(cudaMemcpy(dunpacked_, hdunpacked_, NOPOLS * sizeof(float*), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void**)&dfft_, NOPOLS * sizeof(cufftComplex*)));
    cudaCheckError(cudaMemcpy(dfft_, hdfft_, NOPOLS * sizeof(cufftComplex*), cudaMemcpyHostToDevice));

    scalesamples_ = config_.scaleseconds * 15625;
    dfactors_.resize(scalesamples_);
    thrust::sequence(dfactors_.begin(), dfactors_.end());
    thrust::transform(dfactors_.begin(), dfactors_.end(), dfactors_.begin(), FactorFunctor());
    dmeans_.resize(FILCHANS);
    dstdevs_.resize(FILCHANS);

    pdmeans_ = thrust::raw_pointer_cast(dmeans_.data());
    pdstdevs_ = thrust::raw_pointer_cast(dstdevs_.data());
    pdfactors_ = thrust::raw_pointer_cast(dfactors_.data());

    // NOTE: All the dedispersion done using 8bits input data
    dedispplan_ = unique_ptr<DedispPlan>(new DedispPlan(FILCHANS, samptime_, freqtop_, freqoff_, gpuid_));
    dedispplan_ -> generate_dm_list(config_.dmstart, config_.dmend, 64.0f, 1.10f);
    //dedispextrasamples_ = dedispplan_ -> get_max_delay();
    dedispextrasamples_ = 5000;
    nogulps_ = (int)((dedispgulpsamples_ + dedispextrasamples_ - 1) / dedispgulpsamples_) + 1;;

    // NOTE: Why is this even here?
    // std::this_thread::sleep_for(std::chrono::seconds(10));
    filbuffer_ = unique_ptr<Buffer<unsigned char>>(new Buffer<unsigned char>(gpuid_));
    filbuffer_ -> Allocate(NACCUMULATE, dedispextrasamples_, dedispgulpsamples_, FILCHANS, nogulps_, nostokes_, VDIFLEN * 8 / INBITS / (2 * FILCHANS) / TIMEAVG);

    // STAGE: prepare and launch GPU work
    if (verbose_) {
        PrintSafe("Launching the GPU on pool", poolid_, "...");
    }

    cudaCheckError(cudaGetLastError());

    for (int igstream = 0; igstream < nostreams_; igstream++) {
        gputhreads_.push_back(thread(&GpuPool::DoGpuWork, this, igstream));
    }

    cudaCheckError(cudaStreamCreate(&dedispstream_));
    gputhreads_.push_back(thread(&GpuPool::SendForDedispersion, this, dedispstream_));

    // STAGE: networking
    if (verbose_)
        cout << "Setting up networking..." << endl;

    int netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    sockfiledesc_ = new int[noports_];
    recbufs_ = new unsigned char*[noports_];

    for (int iport = 0; iport < noports_; iport++)
        recbufs_[iport] = new unsigned char[VDIFLEN + HEADLEN];

    ostringstream ssport;
    string strport;

    for (int iport = 0; iport < noports_; iport++) {
        ssport.str("");
        ssport << ports_[iport];
        strport = ssport.str();

        if((netrv = getaddrinfo(strip_[iport].c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
            PrintSafe("getaddrinfo() error:", gai_strerror(netrv), "on pool", poolid_);
        }

        for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
            if((sockfiledesc_[iport] = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
                PrintSafe("Socket error on pool", poolid_);
                continue;
            }

            if(bind(sockfiledesc_[iport], tryme->ai_addr, tryme->ai_addrlen) == -1) {
                close(sockfiledesc_[iport]);
                PrintSafe("Bind error on pool", poolid_);
                continue;
            }
            break;
        }

        if (tryme == NULL) {
            PrintSafe("Failed to bind to the socket", ports_[iport], "on pool", poolid_);
            // NOTE: Bailing like this might not be the best solution, but keep it for now
            exit(EXIT_FAILURE);
        }
    }

    int bufres{4*1024*1024};    // 4MB

    for (int iport = 0; iport < noports_; iport++) {
        if(setsockopt(sockfiledesc_[iport], SOL_SOCKET, SO_RCVBUF, (char *)&bufres, sizeof(bufres)) != 0) {
            PrintSafe("Setsockopt error", errno, "on port", ports_[iport], "on pool", poolid_);
        }
    }

    for (int iport = 0; iport < noports_; iport++)
        receivethreads_.push_back(thread(&GpuPool::ReceiveData, this, iport, ports_[iport]));

}

GpuPool::~GpuPool(void)
{
    for (int ithread = 0; ithread < receivethreads_.size(); ithread++) {
        receivethreads_[ithread].join();
    }

    for (int ithread = 0; ithread < gputhreads_.size(); ithread++) {
        gputhreads_[ithread].join();
    }

//    cudaCheckError(cudaFree(dscaled_));
//    cudaCheckError(cudaFree(dpower_));

/*    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFree(hdscaled_[istoke]));
        cudaCheckError(cudaFree(hdpower_[istoke]));
    }
*/
    cudaCheckError(cudaFree(dfft_));
    cudaCheckError(cudaFree(dunpacked_));
    cudaCheckError(cudaFree(dinpol_));

    for (int ipol = 0; ipol < NOPOLS; ipol++) {
        cudaCheckError(cudaFree(hdunpacked_[ipol]));
        cudaCheckError(cudaFree(hdinpol_[ipol]));
        cudaCheckError(cudaFreeHost(inpol_[ipol]));
    }

    cudaCheckError(cudaFreeHost(inpol_));

    delete [] hdunpacked_;
    delete [] hdinpol_;
    delete [] framenumbers_;
    delete [] readyrawidx_;

}

void GpuPool::DoGpuWork(int stream)
{
    // let us hope one stream will be enough or we will have to squeeze multiple streams into single CPU core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)gpuid_ * 3 + 1, &cpuset);

    int retaff = pthread_setaffinity_np(gputhreads_[stream].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for the GPU processing, stream", stream, "on pool", poolid_);
    }

    if (verbose_) {
        PrintSafe("Starting worker", gpuid_, ":", stream, "on CPU ", sched_getcpu());
    }

    cudaCheckError(cudaSetDevice(gpuid_));

    bool innext{false};
    bool inthis{false};

    int bufferidx;
    int checkidx{5};    // how many receive buffer samples to probe for the existance of the packet
    int startnext;
    int startthis;

    ObsTime frametime;

    unsigned int perframe = VDIFLEN * 8 / INBITS / (2 * FILCHANS) / TIMEAVG;     // the number of output time samples per frame

    int donebuffers = 0;

    float *dscalepower;
    size_t samplesperbuffer = NACCUMULATE * VDIFLEN * 4 / (2 * FILCHANS * TIMEAVG);
    cudaCheckError(cudaMalloc((void**)&dscalepower, FILCHANS * samplesperbuffer * sizeof(float)));

    size_t alreadyscaled = 0;

    while (working_) {
        // TODO: there must be a way to simplify this
        // I should not be allowed to write code
	// startthis check the end of one buffer
	// startnext checks the beginning of the buffer after it
        // TODO: Put the PAF code producer-consumer model
        if (nostreams_ == 1) {      // different behaviour if we are using one stream only - need to check both buffers
            for (int ibuffer = 0; ibuffer < 2; ibuffer++) {
                bufferidx = ibuffer;
                startthis = (ibuffer + 1) * NACCUMULATE - checkidx;
                startnext = ((ibuffer + 1) % 2) * NACCUMULATE; // + checkidx;

                inthis = inthis || readyrawidx_[startthis + checkidx - 1];
                innext = innext || readyrawidx_[startnext + checkidx];

                /*for (int iidx = 0; iidx < checkidx; iidx++) {
                    inthis = inthis && readyrawidx_[startthis + iidx];
                    innext = innext && readyrawidx_[startnext + iidx];
                } */

		if (inthis && innext)
			break;

		inthis = false;
		innext = false;
            }
        } else {
            bufferidx = stream;
            startthis = (stream + 1) * NACCUMULATE - checkidx;
            startnext = ((stream + 1) % nostreams_) * NACCUMULATE; // + checkidx;

            for (int iidx = 0; iidx < checkidx; iidx++) {
                inthis = inthis || readyrawidx_[startthis + iidx];
                innext = innext || readyrawidx_[startnext + iidx];
            }
        }

        if (inthis && innext) {

            readyrawidx_[startthis + checkidx - 1] = false;
            readyrawidx_[startnext + checkidx] = false;

            innext = false;
            inthis = false;
            frametime.startepoch = starttime_.startepoch;
            frametime.startsecond = starttime_.startsecond;
            frametime.framet = framenumbers_[bufferidx / nostreams_ * NACCUMULATE];
            //cout << frametime.framet << endl;
            for (int ipol = 0; ipol < NOPOLS; ipol++) {
                cudaCheckError(cudaMemcpyAsync(hdinpol_[ipol] + stream * rawgpubuffersize_ / nostreams_, inpol_[ipol] + bufferidx * rawgpubuffersize_ / nostreams_, NACCUMULATE * VDIFLEN * sizeof(unsigned char), cudaMemcpyHostToDevice, gpustreams_[stream]));
            }

            UnpackKernelOpt<<<50, 1024, 0, gpustreams_[0]>>>(dinpol_, dunpacked_, NACCUMULATE * VDIFLEN);

            for (int ipol = 0; ipol < NOPOLS; ipol++) {
                cufftCheckError(cufftExecR2C(fftplans_[stream], hdunpacked_[ipol], hdfft_[ipol]));
            }

            if (scaled_) {
                PowerScaleKernelOpt<<<25, FILCHANS, 0, gpustreams_[stream]>>>(dfft_, pdmeans_, pdstdevs_, filbuffer_ -> GetFilPointer(), nogulps_, dedispgulpsamples_, dedispextrasamples_, frametime.framet);
                filbuffer_ -> Update(frametime);
                cudaStreamSynchronize(gpustreams_[stream]);
                cudaCheckError(cudaGetLastError());
            } else {
                PowerKernelOpt<<<25, FILCHANS, 0, gpustreams_[stream]>>>(dfft_, dscalepower);
                cudaStreamSynchronize(gpustreams_[stream]);
                cudaCheckError(cudaGetLastError());
                GetScaleFactorsKernel<<<1, 256, 0, gpustreams_[stream]>>>(dscalepower, pdmeans_, pdstdevs_, pdfactors_, alreadyscaled);
                cudaStreamSynchronize(gpustreams_[stream]);
                cudaCheckError(cudaGetLastError());
                // NOTE: This is not thread safe
                alreadyscaled += 15625;
                if (alreadyscaled >= scalesamples_) {
                    cout << "Scaling factors have been obtained" << endl;
                    cout.flush();
                    scaled_ = true;
                }
            }
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuPool::SendForDedispersion(cudaStream_t dstream)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)gpuid_ * 3 + 2, &cpuset);
    int retaff = pthread_setaffinity_np(gputhreads_[gputhreads_.size() - 1].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for the dedispersion stream on pool", poolid_);
    }

    if (verbose_) {
        PrintSafe("Starting dedispersion thread on CPU", sched_getcpu());
    }

    int gulpssent{0};
    int ready{0};
    header_f filheader;
    filheader.raw_file = "tastytest";
    filheader.source_name = "J1641-45";
    filheader.fch1 = config_.ftop;
    filheader.foff = config_.foff;
    filheader.tsamp = config_.tsamp;
    filheader.az = 0.0;
    filheader.za = 0.0;
    filheader.ra = 0.0;
    filheader.dec = 0.0;
    filheader.rdm = 0.0;
    filheader.ibeam = 1;
    filheader.nbits = 8;
    filheader.nchans = FILCHANS;
    filheader.nifs = 1;
    filheader.data_type = 1;
    filheader.machine_id = 2;
    // NOTE: 5 is for Jodrell
    // TODO: How to deal with other telescopes?
    filheader.telescope_id = 5;

    while(working_) {
        ready = filbuffer_ -> CheckReadyBuffer();
        // TODO: this whole thing with gulpssent breaks down if something is not sent properly and skipped
        // resulting in subsequent time chunk saved with wrong timestamps
        // URGENT TODO: need to sort this out ASAP
        if (ready) {
            filheader.tstart = GetMjd(starttime_.startepoch, starttime_.startsecond + gulpssent * config_.gulp * config_.tsamp);

            if (verbose_) {
                PrintSafe(ready - 1, "filterbank buffer ready for pool", poolid_);
            }
            filbuffer_ -> SendToRam(ready, dedispstream_, (gulpssent % 2));
            filbuffer_ -> SendToDisk((gulpssent % 2), filheader, telescope_, config_.outdir);
            gulpssent++;
        }
    }
}

void GpuPool::HandleSignal(int signum)
{

    cout << "Captured the signal\nWill now terminate!\n";
    working_ = false;
}

void GpuPool::ReceiveData(int portid, int recport)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // TODO: need to test how much we can squeeze out of the single core
    CPU_SET((int)(poolid_) * 3, &cpuset);
    // TODO: pass the thread ID properly
    int retaff = pthread_setaffinity_np(receivethreads_[portid].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        PrintSafe("Error setting thread affinity for receive thread on port", recport, "on pool", poolid_);
    }

    if (verbose_) {
        PrintSafe("Receive thread on port", recport, "running on CPU", sched_getcpu());
    }

    sockaddr_storage theiraddr;
    memset(&theiraddr, 0, sizeof(theiraddr));
    socklen_t addrlen;
    memset(&addrlen, 0, sizeof(addrlen));

    const int pack_per_worker_buf = packperbuf_ / nostreams_;
    int numbytes{0};
    short bufidx{0};
    // this will always be an integer
    unsigned int frameno{0};
    int refsecond{0};
    // thread ID is used to distinguish between polarisations
    int threadid{0};
    int packcount{0};

    // TODO: be careful which port waits
    if (recport == ports_[0]) {
        unsigned char *tempbuf = recbufs_[0];
        // this will wait forever if nothing is sent to the given port
        numbytes = recvfrom(sockfiledesc_[0], recbufs_[0], VDIFLEN + HEADLEN, 0, (struct sockaddr*)&theiraddr, &addrlen);
        starttime_.startepoch = (int)(tempbuf[7] & 0x3f);
        starttime_.startsecond = (int)(tempbuf[0] | (tempbuf[1] << 8) | (tempbuf[2] << 16) | ((tempbuf[3] & 0x3f) << 24));
        telescope_ = string() + (char)tempbuf[13] + (char)tempbuf[12];
        cout << starttime_.startepoch << " " << starttime_.startsecond << endl;
    }

    // NOTE: Waiting for frame 0 to start the recording
    while (working_) {
        // NOTE: This will wait forever if nothing is sent to the given port - recvfrom is a blocking call
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], VDIFLEN * HEADLEN, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            PrintSafe("Error", errno, "on recvfrom on port", recport, "on pool", poolid_);
        }
        if (numbytes == 0)
            continue;

        frameno = (unsigned int)(recbufs_[portid][4] | (recbufs_[portid][5] << 8) | (recbufs_[portid][6] << 16));
        if (frameno == 0) {
            break;

        }
    }

    while(working_) {
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], VDIFLEN + HEADLEN, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            PrintSafe("Error", errno, "on recvfrom on port", recport, "on pool", poolid_);
        }

        if (numbytes == 0)
            continue;

        threadid = (int)(recbufs_[portid][14] | ((recbufs_[portid][15] & 0x02) << 8));
        refsecond = (unsigned int)(recbufs_[portid][0] | (recbufs_[portid][1] << 8) | (recbufs_[portid][2] << 16) | ((recbufs_[portid][3] & 0x3f) << 24));
        frameno = (unsigned int)(recbufs_[portid][4] | (recbufs_[portid][5] << 8) | (recbufs_[portid][6] << 16));	// frame number in this second
        frameno += (refsecond - starttime_.startsecond - 1) * 4000;	// absolute frame number
        //bufidx = (((int)frameno / NACCUMULATE) % nostreams_) * NACCUMULATE + (frameno % NACCUMULATE);
        bufidx = frameno % (rawbuffersize_ / VDIFLEN);
        //cout << "Buffer index: " << bufidx << ":" << threadid <<  ", frame " << frameno << endl;
        //cout.flush();
        framenumbers_[bufidx] = frameno;
        // TODO: Implement a consumer-producer model, similar to the PAF solution
	    std::copy(recbufs_[portid] + HEADLEN, recbufs_[portid] + HEADLEN + VDIFLEN, inpol_[threadid] + VDIFLEN * bufidx);
        if ((threadid == 1) && (frameno > 10))
            readyrawidx_[bufidx] = true;

    }
}
