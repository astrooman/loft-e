#include <algorithm>
#include <bitset>
#include <chrono>
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
                                            dedispgulpsamples_(config.gulp),
                                            fftpoints_(config.fftsize),
                                            filbits_(config.outbits),
                                            filchans_(config.fftsize / config.freqavg),
                                            freqoff_(config.foff),
                                            gpuid_(config.gpuids[id]),
                                            headlen_(config.headlen),
                                            inbits_(config.inbits),
                                            nopols_(config.npol),
                                            nostokes_(config.stokes),
                                            nostreams_(config.streamno),
                                            poolid_(id),
                                            ports_(config.ports[id]),
                                            samptime_(config.tsamp),
                                            scaled_(false),
                                            strip_(config.ips),
                                            telescope_("NT"),
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
    for (int ithread = 0; ithread < receivethreads_.size(); ithread++) {
        receivethreads_[ithread].join();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Cleaning up..." << endl;
        cout << "This may take few seconds..." << endl;
        cout << "Receiving threads done!" << endl;
        cout_guard.unlock();
    }

    for (int ithread = 0; ithread < gputhreads_.size(); ithread++) {
        gputhreads_[ithread].join();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "GPU processing threads done!" << endl;
        cout_guard.unlock();
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

    for (int ipol = 0; ipol < nopols_; ipol++) {
        cudaCheckError(cudaFree(hdunpacked_[ipol]));
        cudaCheckError(cudaFree(hdinpol_[ipol]));
        cudaCheckError(cudaFreeHost(inpol_[ipol]));
    }

    cudaCheckError(cudaFreeHost(inpol_));

    if (verbose_) {
        cout_guard.lock();
        cout << "GPU memory released!" << endl;
        cout_guard.unlock();
    }

    delete [] hdscaled_;
    delete [] hdpower_;
    delete [] hdunpacked_;
    delete [] hdinpol_;
    delete [] framenumbers_;
    delete [] readybufidx_;

    if (verbose_) {
        cout_guard.lock();
        cout << "Host memory released!" << endl;
        cout_guard.unlock();
    }
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
    fftsize_ = accumulate_ * vdiflen_ * nostreams_ * unpackfactor / 2 + 1;
    powersize_ = accumulate_ * vdiflen_ * nostreams_ * unpackfactor;
    // averaging will happen in either power or scale kernel - will be decided later
    scaledsize_ = accumulate_ * vdiflen_ * nostreams_ * unpackfactor / avgfreq_ / avgtime_;

    // fftpoints_ is the number of output frequency channels - need to input 2 times that as we are doing R2C transform
    fftsizes_ = new int[1];
    fftsizes_[0] = 2 * fftpoints_;
    fftbatchsize_ = unpackedsize_ / fftsizes_[0] / nostreams_;
    fftsize_ = fftbatchsize_ * ((int)(fftsizes_[0] / 2) + 1);
    //fftbatchsize_ = fftsize_ / fftsizes_[0] / nostreams_;


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

    hdmeans_ = new float*[nostokes_];
    hdstdevs_ = new float*[nostokes_];

    for (int istoke = 0; istoke < nostokes_; istoke++) {
        cudaCheckError(cudaMalloc((void**)&hdmeans_[istoke], filchans_ * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&hdstdevs_[istoke], filchans_ * sizeof(float)));
    }

    cudaCheckError(cudaMalloc((void**)&dmeans_, nostokes_ * sizeof(float*)));
    cudaCheckError(cudaMalloc((void**)&dstdevs_, nostokes_ * sizeof(float*)));

    cudaCheckError(cudaMemcpy(dmeans_, hdmeans_, nostokes_ * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(dstdevs_, hdstdevs_, nostokes_ * sizeof(float*), cudaMemcpyHostToDevice));

    // TODO: this should really be template-like - we may choose to scale to different number of bits
    dedispplan_ = unique_ptr<DedispPlan>(new DedispPlan(filchans_, samptime_, freqtop_, freqoff_, gpuid_));
    dedispplan_ -> generate_dm_list(config_.dstart, config_.dend, 64.0f, 1.10f);
    //dedispextrasamples_ = dedispplan_ -> get_max_delay();
    dedispextrasamples_ = 5000;

    nogulps_ = (int)((dedispgulpsamples_ + dedispextrasamples_ - 1) / dedispgulpsamples_) + 1;;

    std::this_thread::sleep_for(std::chrono::seconds(10));
    if (verbose_)
        cout << "Will create a buffer for the total of " << nogulps_ * dedispgulpsamples_ + dedispextrasamples_ << " time samples" << endl;

    filbuffer_ = unique_ptr<Buffer>(new Buffer(gpuid_));
    filbuffer_ -> Allocate(accumulate_, dedispextrasamples_, dedispgulpsamples_, filchans_, nogulps_, nostokes_, vdiflen_ * 8 / inbits_ / (2 * fftpoints_) / avgtime_, filbits_);

    // STAGE: prepare and launch GPU work
    if (verbose_)
        cout << "Launching the GPU..." << endl;

    gpustreams_ = new cudaStream_t[nostreams_];
    fftplans_ = new cufftHandle[nostreams_];
    //fftsizes_ = new int[1];
    // fftpoints_ is the number of output frequency channels - need to input 2 times that as we are doing R2C transform
    //fftsizes_[0] = 2 * fftpoints_;
    //fftbatchsize_ = unpackedsize_ / fftsizes_[0] / nostreams_;
    //fftbatchsize_ = fftsize_ / fftsizes_[0] / nostreams_;

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
    int needblocks = (needthreads - 1) / cudathreads_[0] + 1;

    cudablocks_[0] = min(needblocks, 65536);
    rem_ = needthreads - cudablocks_[0] * cudathreads_[0];

    perblock_ = 128;        // the number of OUTPUT time samples (after averaging) per block
    // NOTE: Have to be very careful with this number as vdiflen is not a power of 2
    // This will cause problems when accumulate_ * 8 / inbits is less than 1 / (avgtime_ * perblock_ * fftsize_[0]
    // This will give a non-integer number of blocks
    cudablocks_[1] = accumulate_ * vdiflen_ * 8 / inbits_ / avgtime_ / perblock_ / fftsizes_[0];
    cudathreads_[1] = filchans_;        // each thread in the block will output a single AVERAGED frequency channel

    cout << "Unpack kernel grid: " << cudablocks_[0] << " blocks and " << cudathreads_[0] << " threads" <<endl;
    cout << "Power kernel grid: " << cudablocks_[1] << " blocks and " << cudathreads_[1] << " threads" <<endl;

    switch (filbits_) {
        case 8:
            {
                for (int igstream = 0; igstream < nostreams_; igstream++) {
                    gputhreads_.push_back(thread(&GPUpool::DoGpuWork<unsigned char>, this, igstream));
                }
            }
            break;
        case 16:
            {
                for (int igstream = 0; igstream < nostreams_; igstream++) {
                    gputhreads_.push_back(thread(&GPUpool::DoGpuWork<unsigned short>, this, igstream));
                }
            }
            break;
        case 32:
            {
                 for (int igstream = 0; igstream < nostreams_; igstream++) {
                     gputhreads_.push_back(thread(&GPUpool::DoGpuWork<float>, this, igstream));
                 }
            }
            break;
        default:
            {
                cerr << "Unsupported number of output bits!\n";
                cerr << "Will set to 8!";
                for (int igstream = 0; igstream < nostreams_; igstream++) {
                    gputhreads_.push_back(thread(&GPUpool::DoGpuWork<unsigned char>, this, igstream));
                }
            }
    }

    cudaCheckError(cudaStreamCreate(&dedispstream_));
    gputhreads_.push_back(thread(&GPUpool::SendForDedispersion, this, dedispstream_));

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

void GPUpool::SendForDedispersion(cudaStream_t dstream)
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
    filheader.ibeam = 1;
    filheader.nbits = filbits_;
    filheader.nchans = filchans_;
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
            if (scaled_) {
                filheader.tstart = GetMjd(starttime_.startepoch, starttime_.startsecond + gulpssent * config_.gulp * config_.tsamp);

                if (verbose_) {
                    cout_guard.lock();
                    cout << ready - 1 << " filterbank buffer ready" << endl;
                    cout_guard.unlock();
                }
                filbuffer_ -> SendToRam(ready, dedispstream_, (gulpssent % 2));
                filbuffer_ -> SendToDisk((gulpssent % 2), filheader, telescope_, config_.outdir);
                gulpssent++;
            } else {    // the first buffer will be used for getting the scaling factors
                switch(filbits_) {
                    case 8:
                        filbuffer_ -> GetScaling<unsigned char>(ready, dedispstream_, dmeans_, dstdevs_);
                        break;
                    case 16:
                        filbuffer_ -> GetScaling<unsigned short>(ready, dedispstream_, dmeans_, dstdevs_);
                        break;
                    case 32:
                        filbuffer_ -> GetScaling<float>(ready, dedispstream_, dmeans_, dstdevs_);
                        break;
                    // NOTE: This should never be executed
                    default:
                        break;
                }
                scaled_ = true;
                ready = 0;
                if (verbose_) {
                    cout_guard.lock();
                    cout << "Scaling factors have been obtained" << endl;
                    cout_guard.unlock();
                }
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
        // this will wait forever if nothing is sent to the given port
        numbytes = recvfrom(sockfiledesc_[0], recbufs_[0], vdiflen_ + headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen);
        starttime_.startepoch = (int)(tempbuf[7] & 0x3f);
        starttime_.startsecond = (int)(tempbuf[0] | (tempbuf[1] << 8) | (tempbuf[2] << 16) | ((tempbuf[3] & 0x3f) << 24));
        telescope_ = string() + (char)tempbuf[13] + (char)tempbuf[12];
        cout << starttime_.startepoch << " " << starttime_.startsecond << endl;
    }

    while (working_) {
        // this will wait forever if nothing is sent to the given port
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], vdiflen_ * headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << recport << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;

        frameno = (unsigned int)(recbufs_[portid][4] | (recbufs_[portid][5] << 8) | (recbufs_[portid][6] << 16));
        //cout << "Starting frame: " << frameno << endl;
        //cout.flush();
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
        //cout << "Buffer index: " << bufidx << ":" << threadid <<  ", frame " << frameno << endl;
        //cout.flush();
        framenumbers_[bufidx] = frameno;
	    std::copy(recbufs_[portid] + headlen_, recbufs_[portid] + headlen_ + vdiflen_, inpol_[threadid] + vdiflen_ * bufidx);
        if ((threadid == 1) && (frameno > 10))
            readybufidx_[bufidx] = true;

    }
}
