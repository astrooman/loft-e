#include <algorithm>
#include <bitset>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <boost/array.hpp>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <cufft.h>
#include <cuda.h>
#include <pthread.h>
#include <thrust/device_vector.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/dedisp.hpp"
#include "dedisp/DedispPlan.hpp"
#include "errors.hpp"
#include "filterbank.hpp"
#include "get_mjd.hpp"
#include "heimdall/pipeline.hpp"
#include "kernels.cuh"
#include "obs_time.hpp"
#include "pdif.hpp"
#include "pool_multi.cuh"

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
using std::mutex;
using std::ostringstream;
using std::pair;
using std::queue;
using std::string;
using std::thread;
using std::unique_ptr;
using std::vector;

#define HEADER 32

mutex cout_guard;

/* ########################################################
TODO: Too many copies - could I use move in certain places?
#########################################################*/

/*####################################################
IMPORTANT: from what I seen in the system files:
There is only one NUMA node.
6 (sic!) physical cores
####################################################*/

int power2factor(unsigned int inbytes) {
    if ((inbytes % 2) != 0)
        return 1;      // don't even  bother with odd numbers

    int factor = 4;

    while ((inbytes % factor) == 0) {
        factor *= 2;
    }

    return factor / 2;
};

Oberpool::Oberpool(InConfig config) : ngpus(config.ngpus)
{

    for (int ii = 0; ii < ngpus; ii++) {
        gpuvector.push_back(unique_ptr<GPUpool>(new GPUpool(ii, config)));
    }

    for (int ii = 0; ii < ngpus; ii++) {
        threadvector.push_back(thread(&GPUpool::Initialise, std::move(gpuvector[ii])));
    }

}

Oberpool::~Oberpool(void)
{
    for (int ii = 0; ii < ngpus; ii++) {
        threadvector[ii].join();
    }

}

bool GPUpool::working_ = true;

GPUpool::GPUpool(int id, InConfig config) : accumulate_(config.accumulate),
                                            avgfreq_(config.freqavg),
                                            avgtime_(config.timesavg),
                                            dedispgulp_(config.gulp),
                                            fftpoints_(config.fftsize),
                                            filchans_(config.fftsize / config.freqavg),
                                            gpuid_(config.gpuids[id]),
                                            headlen_(config.headlen),
                                            inbits_(config.inbits),
                                            nopols_(config.npol),
                                            nostokes_(config.stokes),
                                            nostreams_(config.streamno),
                                            poolid_(id),
                                            ports_(config.ports[id]),
                                            scaled_(false),
                                            strip_(config.ips),
                                            vdiflen_(config.vdiflen),
                                            verbose_(config.verbose)
{
    availthreads_ = min(nostreams_ + 1, thread::hardware_concurrency());

    config_ = config;

    if (verbose_) {
        cout_guard.lock();
        cout << "Starting GPU pool " << gpuid_ << endl;
        cout << "This may take few seconds..." << endl;
	    cout.flush();
        cout_guard.unlock();
    }
}

GPUpool::~GPUpool(void)
{
    // TODO: join the processing threads

    for (int ithread = 0; ithread < receivethreads_.size(); ithread++) {
        receivethreads_[ithread].join();
    }

    for (int ithread = 0; ithread < gputhreads_.size(); ithread++) {
        gputhreads_[ithread].join();
    }

    cudaCheckError(cudaFree(dscaled_));
    cudaCheckError(cudaFree(dpower_));

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaFree(hdscaled_[istoke]));
        cudaCheckError(cudaFree(hdpower_[istoke]));
    }

    cudaCheckError(cudaFree(dfft_));
    cudaCheckError(cudaFree(dunpacked_));
    cudaCheckError(cudaFree(dinpol_));

    for (int ipol = 0; ipol < nopols_; ipol++) {
        cudaCheckError(cudaFree(hdunpacked_[ipol]));
        cudaCheckError(cudaFree(hdinpol_[ipol]));
        cudaCheckError(cudaFreeHost(inpol_[ipol]));
    }

    cudaCheckError(cudaFreeHost(inpol_));

    delete [] hdscaled_;
    delete [] hdpower_;
    delete [] hdunpacked_;
    delete [] hdinpol_;
    delete [] framenumbers_;
    delete [] readybufidx_;
}

void GPUpool::Initialise(void)
{

    noports_ = ports_.size();

    signal(SIGINT, GPUpool::HandleSignal);
    cudaCheckError(cudaSetDevice(gpuid_));

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // * 3 as there are 3 cores available for each telescope/GPU
    CPU_SET((int)(gpuid_) * 3, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for the GPU pool " << gpuid_ << endl;
        cout_guard.unlock();
    }

    if(verbose_) {
        cout_guard.lock();
        cout << "GPU pool for device " << gpuid_ << " running on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
    }

    // STAGE: memory
    if (verbose_)
        cout << "Initialising the memory..." << endl;


    inpolbufsize_ = max(2, nostreams_) * accumulate_ * vdiflen_;		// only one-stream scenario will have an extra buffer
    inpolgpusize_ = accumulate_ * vdiflen_ * nostreams_;
    int unpackfactor = 8 / inbits_;
    unpackedsize_ = accumulate_ * vdiflen_ * nostreams_ * unpackfactor;
    fftsize_ = accumulate_ * vdiflen_ * nostreams_ * unpackfactor;
    powersize_ = accumulate_ * vdiflen_ * nostreams_ * unpackfactor;
    // averaging will happen in either power or scale kernel - will be decided later
    scaledsize_ = accumulate_ * vdiflen_ * nostreams_ * unpackfactor / avgfreq_ / avgtime_;

    readybufidx_ = new bool[inpolbufsize_];
    framenumbers_ = new unsigned int[max(2, nostreams_) * accumulate_];

    hdinpol_ = new unsigned char*[nopols_];
    hdunpacked_ = new float*[nopols_];
    hdfft_ = new cufftComplex*[nopols_];
//    hdpower_ = new float*[nostokes_];
//    hdscaled_ = new unsigned char*[nostokes_];
    cudaCheckError(cudaHostAlloc((void**)&inpol_, nopols_ * sizeof(unsigned char*), cudaHostAllocDefault));

    cout << sizeof(cufftComplex) << endl;

    for (int ipol = 0; ipol < nopols_; ipol++) {
        cudaCheckError(cudaHostAlloc((void**)&inpol_[ipol], inpolbufsize_ * sizeof(unsigned char), cudaHostAllocDefault));
        cudaCheckError(cudaMalloc((void**)&hdinpol_[ipol], inpolgpusize_ * sizeof(unsigned char)));
        cudaCheckError(cudaMalloc((void**)&hdunpacked_[ipol], unpackedsize_ * sizeof(float)));      // remember we are unpacking to float
        cudaCheckError(cudaMalloc((void**)&hdfft_[ipol], fftsize_ * sizeof(cufftComplex)));
    }

    cudaCheckError(cudaMalloc((void**)&dinpol_, nopols_ * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dinpol_, hdinpol_, nopols_ * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void**)&dunpacked_, nopols_ * sizeof(float*)));
    cudaCheckError(cudaMemcpy(dunpacked_, hdunpacked_, nopols_ * sizeof(float*), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void**)&dfft_, nopols_ * sizeof(cufftComplex*)));
    cudaCheckError(cudaMemcpy(dfft_, hdfft_, nopols_ * sizeof(cufftComplex*), cudaMemcpyHostToDevice));

/*    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&hdpower_[istoke], powersize_ * sizeof(float)));
        // TODO: this should really be template-like - we may choose to scale to different number of bits
        cudaCheckError(cudaMalloc((void**)&hdscaled_[istoke], scaledsize_ * sizeof(unsigned char)));
    }
*/
//    cudaCheckError(cudaMalloc((void**)&dpower_, nostokes_ * sizeof(float*)));
//    cudaCheckError(cudaMemcpy(dpower_, hdpower_, nostokes_ * sizeof(float*), cudaMemcpyHostToDevice));

//    cudaCheckError(cudaMalloc((void**)&dscaled_, nostokes_ * sizeof(unsigned char*)));
//    cudaCheckError(cudaMemcpy(dscaled_, hdscaled_, nostokes_ * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    hdmeans_ = new float*[stokes_];
    hdstdevs_ = new float*[stokes_];

    for (int istoke = 0; istoke < stokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&hdmeans[istoke], filchans_ * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&hdstdevs[istoke]. filchans * sizeof(float)));
    }

    // TODO: this should really be template-like - we may choose to scale to different number of bits
    dedispplan_ = unique_ptr<DedispPlan>(new DedispPlan(filchans_, samptime_, freqtop_, ffreqoff_, gpuid));
    dedispplan_ -> generate_dm_list(config_.dstart, config_.dend, 64.0f, 1.10f);
    dedispextrasamples_ = dedispplan_ -> get_max_delay();

    nogulps_ = (int)((dedispgulpsamples_ + dedispextrasamples_ - 1) / dedispgulpsamples_) + 1;;

    filbuffer_ = unique_ptr<Buffer<unsigned char>>(new Buffer<unsigned char>(gpuid_));
    filbuffer_ -> allocate(accumulate_, dedispextrasamples_, dedispgulpsamples_, filchans_, nogulps_, nostokes_);

    // STAGE: prepare and launch GPU work
    if (verbose_)
        cout << "Launching the GPU..." << endl;

    gpustreams_ = new cudaStream_t[nostreams_];
    fftplans_ = new cufftHandle[nostreams_];
    fftsizes_ = new int[1];
    // fftpoints_ is the number of output frequency channels - need to input 2 times that as we are doing R2C transform
    fftsizes_[0] = 2 * fftpoints_;
    fftbatchsize_ = fftsize_ / fftsizes_[0];

    cout << fftsizes_[0] << " " << fftsize_ << " " << fftbatchsize_ << endl;

    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cudaCheckError(cudaStreamCreate(&gpustreams_[igstream]));
        cufftCheckError(cufftPlanMany(&fftplans_[igstream], 1, fftsizes_, NULL, 1, fftpoints_, NULL, 1, fftpoints_, CUFFT_R2C, fftbatchsize_));
        cufftCheckError(cufftSetStream(fftplans_[igstream], gpustreams_[igstream]));
    }

    int nokernels = 3;  // unpack, power and scale
    cudablocks_ = new unsigned int[nokernels];
    cudathreads_ = new unsigned int[nokernels];

    // currently limit set to 1024, but can be lowered down, depending on the results of the tests
    sampperthread_ = min(power2factor(accumulate_ * vdiflen_), 1024);
    int needthreads = accumulate_ * vdiflen_ / sampperthread_;
    cudathreads_[0] = min(needthreads, 1024);
    int needblocks = needthreads / cudathreads_[0];
    cudablocks_[0] = min(needblocks, 65536);
    rem_ = needthreads - cudablocks_[0] * cudathreads_[0];

    perblock_ = 128;        // the number of OUTPUT time samples (after averaging) per block
    cudablocks_[1] = accumulate_ * vdiflen_ * 8 / inbits_ / avgtime_ / perblock_;
    cudathreads_[1] = filchans_;        // each thread in the block will output a single AVERAGED frequency channel


    for (int igstream = 0; igstream < nostreams_; igstream++) {
        gputhreads_.push_back(thread(&GPUpool::DoGpuWork, this, igstream));
    }

    cudaCheckError(cudaStreamCreate(&dedispstream_));
    gputhreads_.push_back(thread(&GPUpool::SendForDedispersion, this dedispstream_));

    // STAGE: networking
    if (verbose_)
        cout << "Setting up networking..." << endl;

    int netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    cout << noports_ << endl;

    sockfiledesc_ = new int[noports_];
    recbufs_ = new unsigned char*[noports_];

    cout << strip_[0] << endl;
    cout << vdiflen_ << " " << headlen_ << endl;

    for (int iport = 0; iport < noports_; iport++)
        recbufs_[iport] = new unsigned char[vdiflen_ + headlen_];

    ostringstream ssport;
    string strport;

    for (int iport = 0; iport < noports_; iport++) {
        cout << strip_[iport] << endl;
        cout << strip_[iport].c_str() << endl;
        cout.flush();
        ssport.str("");
        ssport << ports_[iport];
        strport = ssport.str();

        cout << strip_[iport] << endl;
        cout << strip_[iport].c_str() << endl;

        if((netrv = getaddrinfo(strip_[iport].c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
            cout_guard.lock();
            cerr <<  "getaddrinfo() error: " << gai_strerror(netrv) << endl;
            cout_guard.unlock();
        }

        for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
            if((sockfiledesc_[iport] = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
                cout_guard.lock();
                cerr << "Socket error\n";
                cout_guard.unlock();
                continue;
            }

            if(bind(sockfiledesc_[iport], tryme->ai_addr, tryme->ai_addrlen) == -1) {
                close(sockfiledesc_[iport]);
                cout_guard.lock();
                cerr << "Bind error\n";
                cout_guard.unlock();
                continue;
            }
            break;
        }

        if (tryme == NULL) {
            cout_guard.lock();
            cerr << "Failed to bind to the socket " << ports_[iport] << "\n";
            cout_guard.unlock();
        }
    }

    int bufres{4*1024*1024};    // 4MB

    for (int iport = 0; iport < noports_; iport++) {
        if(setsockopt(sockfiledesc_[iport], SOL_SOCKET, SO_RCVBUF, (char *)&bufres, sizeof(bufres)) != 0) {
            cout_guard.lock();
            cerr << "Setsockopt error on port " << ports_[iport] << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
    }

    for (int iport = 0; iport < noports_; iport++)
        receivethreads_.push_back(thread(&GPUpool::ReceiveData, this, iport, ports_[iport]));

}

void GPUpool::DoGpuWork(int stream)
{
    // let us hope one stream will be enough or we will have to squeeze multiple streams into single CPU core
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)gpuid_ * 3 + 1, &cpuset);

    int retaff = pthread_setaffinity_np(gputhreads_[stream].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        cerr << "Error setting thread affinity for the GPU processing, stream " << stream << endl;
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Starting worker " << gpuid_ << ":" << stream << " on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
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

    while (working_) {

        // TODO: there must be a way to simplify this
        // I should not be allowed to write code
        if (nostreams_ == 1) {      // different behaviour if we are using one stream only - need to check both buffers
            for (int ibuffer = 0; ibuffer < 2; ibuffer++) {
                bufferidx = ibuffer;
                startthis = (ibuffer + 1) * accumulate_ - 1 - checkidx;
                startnext = (ibufer + 1) % 2) * accumulate_ + checkidx;

                for (int iidx = 0; iidx < checkidx; iidx++) {
                    inthis = inthis || readybufidx_[startthis + iidx];
                    innext = innext || readybufidx_[startnext + iidx];
                }
            }
        } else {
            bufferidx = streamno;
            starthis = (stream + 1) * accumulate_ - 1 - checkidx;
            startnext = ((stream + 1) % nostreams_) * accumulate + checidx;

            for (int iidx = 0; iidx < checkidx; iidx++) {
                inthis = inthis || readybufidx_[startthis + iidx];
                innext = innext || readybufidx_[startnext + iidx];
            }
        }

        if (inthis && innext) {

            innext = false;
            inthis = false;
            frametime.startepoch = starttime_.startepoch;
            frametime.startsecond = starttime_.startsecond;
            frametime.framet = framenumbers_[bufferidx / nostreams_ * accumulate_];

            for (int ipol = 0; ipol < nopols_; ipol++) {
                cudaCheckError(cudaMemcpyAsync(hdinpol_[ipol] + stream * inpolgpusize_ / nostreams_, inpol_[ipol] + bufferidx * inpolgpusize_ / nostreams_, accumulate_ * vdiflen_ * sizeof(unsigned char), cudaMemcpyHostToDevice, gpustreams_[stream]));
            }
            UnpackKernel<<<cudablocks_[0], cudathreads_[0], 0, gpustreams_[stream]>>>(dinpol_, dunpacked_, nopols_, sampperthread_, rem_, accumulate_ * vdiflen_, 8 / inbits_);
            for (int ipol = 0; ipol < nopols; ipol++) {
                cufftCheckError(cufftExecR2C(fftplans_[stream], dunpacked_[ipol] + stream * unpackedsize_, dfft_[ipol] + stream * fftsize_));
            }
            //PowerScaleKernel<<<cudablocks_[1], cudathreads_[1], 0, gpustreams_[stream]>>>(dfft_, dscaled_, dmeans_, dstdevs_, avgfreq_, avgtime_, filchans_, perblock_, stream * fftsize_, stream * scaledsize_);
            // this version should really be used, as we want to save directly into the filterbank buffer
            PowerScaleKernel<<<cudablocks_[1], cudathreads_[1], 0, gpustreams_[stream]>>>(dfft_, filbuffer_ -> GetFilPointer(), dmeans_, dstdevs_, avgfreq_, avgtime_, filchans_, perblock_, stream * fftsize_, nogulps_, gulpsamples_, extrasamples_, frametime.framet, perframe);
            filbuffer_ -> Update(frametime);
            // ScaleKernel<<<cudablocks_[2], cudathreads_[2], 0, gpustreams_[stream]>>>(dpower_, filbuffer_ -> GetFilPointer());
        } else {
            std::this_thread::yield();
        }
    }


}

void GPUpool::SendForDedispersion(int dstream)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)gpuid_ * 3 + 2, &cpuset);
    int retaff = pthread_setaffinity_np(gputhreads_[gputhreads_.size() - 1].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for the dedispersion stream" << endl;
        cout_guard.unlock();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Starting dedispersion thread on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
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
    filheader.beamno = 1;
    filheader.nbits = 8;
    filheader.nchans = filchans_;
    filheader.nifs = 1;
    filheader.data_type = 1;
    filheader.machine_id = 2;
    filheader.telescope_id = 2;

    while(working_) {
        ready = filbuffer_ -> CheckReadyBuffer();
        // TODO: this whole thing with gulpssent breaks down if something is not sent properly and skipped
        // resulting in subsequent time chunk saved with wrong timestamps
        // URGENT TODO: need to sort this out ASAP
        if (ready)
            if (scaled_) {
                filheader.tstat = get_mjd(starttime_.startepoch, starttime_.startsecond + gulpssent * config_.gulp * config.tsamp_);

                if (verbose_) {
                    cout_guard.lock();
                    cout << ready - 1 << " filterbank buffer ready" << endl;
                    cout_guard.unlock();
                }
                filbuffer_ -> SendToRam(ready, dedispstream_, (gulpssent % 2));
                filbuffer_ -> SendToDisk((gulpssent % 2), filheader, config_.outdir);
                gulpssent++;
            } else {    // the first buffer will be used for getting the scaling factors
                scaled_ = true;
                ready = 0;
                if (verbose_) {
                    filbuffer_ -> GetScaling(ready, dedispstream_, dmeans_, dstdevs_)
                    cout_guard.lock();
                    cout << "Scaling factors have been obtained" << endl;
                    cout_guard.unlock();
                }
            }
    }
}

void GPUpool::HandleSignal(int signum)
{

    cout << "Captured the signal\nWill now terminate!\n";
    working_ = false;
}

void GPUpool::ReceiveData(int portid, int recport)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // TODO: need to test how much we can squeeze out of the single core
    // might depend on portid if can't pack everything in one core
    CPU_SET((int)(poolid_) * 3, &cpuset);
    // TODO: pass the thread ID properly
    int retaff = pthread_setaffinity_np(receivethreads_[portid].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for receive thread on port " << recport << endl;
        cout_guard.unlock();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Receive thread on port " << recport << " running on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
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
        numbytes = recvfrom(sockfiledesc_[0], recbufs_[0], vdiflen_ + headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen);
        starttime_.startepoch = (int)(tempbuf[4] & 0x3f);
        starttime_.startsecond = (int)(tempbuf[3] | (tempbuf[2] << 8) | (tempbuf[1] << 16) | ((tempbuf[0] & 0x3f) << 24));
    }

    while (working_) {
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], vdiflen_ * headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << recport << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;
        frameno = (unsigned int)(recbufs_[portid][4] | (recbufs_[portid][5] << 8) | (recbufs_[portid][6] << 16));
        cout << frameno << endl;
        cout.flush();
        if (frameno == 0) {
            break;
        } // wait until reaching frame zero of next second before beginnning recording
    }

    while(working_) {
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], vdiflen_ + headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << recport << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;

        threadid = (int)(recbufs_[portid][14] | ((recbufs_[portid][15] & 0x02) << 8));
        refsecond = (unsigned int)(recbufs_[portid][0] | (recbufs_[portid][1] << 8) | (recbufs_[portid][2] << 16) | ((recbufs_[portid][3] & 0x3f) << 24));
        frameno = (unsigned int)(recbufs_[portid][4] | (recbufs_[portid][5] << 8) | (recbufs_[portid][6] << 16));	// frame number in this second
        frameno += (refsecond - starttime_.startsecond - 1) * 4000;	// absolute frame number
        //bufidx = (((int)frameno / accumulate_) % nostreams_) * accumulate_ + (frameno % accumulate_);
        bufidx = frameno % (inpolbufsize_ / vdiflen_);
        framenumbers_[bufidx] = frameno;
	    std::copy(recbufs_[portid] + headlen_, recbufs_[portid] + headlen_ + vdiflen_, inpol_[threadid] + vdiflen_ * bufidx);
        readybufidx_[bufidx] = true;
        //}
    }
}
