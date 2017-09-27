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

#define HEADER 32

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



bool GpuPool::working_ = true;

GpuPool::GpuPool(int id, InConfig config) : accumulate_(config.accumulate),
                                            avgfreq_(config.freqavg),
                                            avgtime_(config.timeavg),
                                            config_(config),
                                            dedispgulpsamples_(config.gulp),
                                            fftpoints_(config.fftsize),
                                            filchans_(config.fftsize / config.freqavg),
                                            freqoff_(config.foff),
                                            gpuid_(config.gpuids[id]),
                                            headlen_(config.headlen),
                                            inbits_(config.inbits),
                                            nopols_(config.nopols),
                                            nostokes_(config.nostokes),
                                            nostreams_(config.nostreams),
                                            poolid_(id),
                                            ports_(config.ports[id]),
                                            samptime_(config.tsamp),
                                            scaled_(false),
                                            strip_(config.ips),
                                            telescope_("NT"),
                                            vdiflen_(config.vdiflen),
                                            verbose_(config.verbose)
{
    availthreads_ = thread::hardware_concurrency();
    if (availthreads_ == 0) {
        cerr << "Could not obtain the number of cores on node " << poolid_ << "!\n";
        cerr << "Will set to 6!";
        // NOTE: This is true for LOFT-e machines for now - be careful as it might change in the future
        availthreads_ = 6;
    }

    if (verbose_) {
        PrintSafe("Starting the GPU pool", gpuid_, "...");
    }
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

    for (int ipol = 0; ipol < nopols_; ipol++) {
        cudaCheckError(cudaFree(hdunpacked_[ipol]));
        cudaCheckError(cudaFree(hdinpol_[ipol]));
        cudaCheckError(cudaFreeHost(inpol_[ipol]));
    }

    cudaCheckError(cudaFreeHost(inpol_));

    delete [] hdunpacked_;
    delete [] hdinpol_;
    delete [] framenumbers_;
    delete [] readybufidx_;

}

void GpuPool::Initialise(void)

{
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
    inpolbufsize_ = 2 * accumulate_ * vdiflen_;
    readybufidx_ = new bool[inpolbufsize_];
    framenumbers_ = new unsigned int[2 * accumulate_];
    inpolgpusize_ = accumulate_ * vdiflen_ * nostreams_;
    // NOTE: 4 unpacked samples per incoming byte
    unpackedsize_ = accumulate_ * vdiflen_ * nostreams_ * 4;

    fftsizes_ = new int[1];
    // NOTE: Need twice as many input samples for the R2C transform
    fftsizes_[0] = 2 * fftpoints_;
    fftbatchsize_ = unpackedsize_ / fftsizes_[0] / nostreams_;
    // NOTE: The output of the R2C transform is FFTSIZE / 2 + 1
    fftsize_ = fftbatchsize_ * ((int)fftsizes_[0] / 2 + 1) * nostreams_;

    fftplans_ = new cufftHandle[nostreams_];
    for (int igstream = 0; igstream < nostreams_; igstream++) {
        cudaCheckError(cudaStreamCreate(&gpustreams_[igstream]));
        cufftCheckError(cufftPlanMany(&fftplans_[igstream], 1, fftsizes_, NULL, 1, fftpoints_, NULL, 1, fftpoints_, CUFFT_R2C, fftbatchsize_));
        cufftCheckError(cufftSetStream(fftplans_[igstream], gpustreams_[igstream]));
    }

    hdinpol_ = new unsigned char*[nopols_];
    hdunpacked_ = new float*[nopols_];
    hdfft_ = new cufftComplex*[nopols_];

    cudaCheckError(cudaHostAlloc((void**)&inpol_, nopols_ * sizeof(unsigned char*), cudaHostAllocDefault));

    for (int ipol = 0; ipol < nopols_; ipol++) {
        cudaCheckError(cudaHostAlloc((void**)&inpol_[ipol], inpolbufsize_ * sizeof(unsigned char), cudaHostAllocDefault));
        cudaCheckError(cudaMalloc((void**)&hdinpol_[ipol], inpolgpusize_ * sizeof(unsigned char)));
        // NOTE: We are unpacking 2-bit number to floats - courtesy of cuFFT
        cudaCheckError(cudaMalloc((void**)&hdunpacked_[ipol], unpackedsize_ * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&hdfft_[ipol], fftsize_ * sizeof(cufftComplex)));
    }

    cudaCheckError(cudaMalloc((void**)&dinpol_, nopols_ * sizeof(unsigned char*)));
    cudaCheckError(cudaMemcpy(dinpol_, hdinpol_, nopols_ * sizeof(unsigned char*), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void**)&dunpacked_, nopols_ * sizeof(float*)));
    cudaCheckError(cudaMemcpy(dunpacked_, hdunpacked_, nopols_ * sizeof(float*), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMalloc((void**)&dfft_, nopols_ * sizeof(cufftComplex*)));
    cudaCheckError(cudaMemcpy(dfft_, hdfft_, nopols_ * sizeof(cufftComplex*), cudaMemcpyHostToDevice));

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

    // NOTE: All the dedispersion done using 8bits input data
    dedispplan_ = unique_ptr<DedispPlan>(new DedispPlan(filchans_, samptime_, freqtop_, freqoff_, gpuid_));
    dedispplan_ -> generate_dm_list(config_.dmstart, config_.dmend, 64.0f, 1.10f);
    //dedispextrasamples_ = dedispplan_ -> get_max_delay();
    dedispextrasamples_ = 5000;
    nogulps_ = (int)((dedispgulpsamples_ + dedispextrasamples_ - 1) / dedispgulpsamples_) + 1;;

    // NOTE: Why is this even here?
    // std::this_thread::sleep_for(std::chrono::seconds(10));
    filbuffer_ = unique_ptr<Buffer<unsigned char>>(new Buffer<unsigned char>(gpuid_));
    filbuffer_ -> Allocate(accumulate_, dedispextrasamples_, dedispgulpsamples_, filchans_, nogulps_, nostokes_, vdiflen_ * 8 / inbits_ / (2 * fftpoints_) / avgtime_);

    // STAGE: prepare and launch GPU work
    if (verbose_) {
        PrintSafe("Launching the GPU on pool", poolid_, "...");
    }

    // NOTE: The block and thread calculations will be unnecessary with the optimised code
    int nokernels = 3;  // unpack, power and scale
    cudablocks_ = new unsigned int[nokernels];
    cudathreads_ = new unsigned int[nokernels];

    // currently limit set to 1024, but can be lowered down, depending on the results of the tests
    sampperthread_ = min(power2factor(accumulate_ * vdiflen_), 1024);
    int needthreads = accumulate_ * vdiflen_ / sampperthread_;
    cudathreads_[0] = min(needthreads, 1024);
    int needblocks = (needthreads - 1) / cudathreads_[0] + 1;

    cudablocks_[0] = min(needblocks, 65536);

    perblock_ = 100;        // the number of OUTPUT time samples (after averaging) per block
    // NOTE: Have to be very careful with this number as vdiflen is not a power of 2
    // This will cause problems when accumulate_ * 8 / inbits is less than 1 / (avgtime_ * perblock_ * fftsize_[0]
    // This will give a non-integer number of blocks
    cudablocks_[1] = accumulate_ * vdiflen_ * 8 / inbits_ / avgtime_ / perblock_ / fftsizes_[0];
    cudathreads_[1] = filchans_;        // each thread in the block will output a single AVERAGED frequency channel

    // NOTE: This should only be used for debug purposes
    //cout << "Unpack kernel grid: " << cudablocks_[0] << " blocks and " << cudathreads_[0] << " threads" <<endl;
    //cout << "Power kernel grid: " << cudablocks_[1] << " blocks and " << cudathreads_[1] << " threads" <<endl;

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
                // cudaCheckError(cudaMemcpyAsync(hdinpol_[ipol] + stream * inpolgpusize_ / nostreams_, inpol_[ipol] + bufferidx * inpolgpusize_ / nostreams_, accumulate_ * vdiflen_ * sizeof(unsigned char), cudaMemcpyHostToDevice, gpustreams_[stream]));
                cudaCheckError(cudaMemcpyAsync(hdinpol_[ipol] + stream * inpolgpusize_ / nostreams_, inpol_[ipol] + bufferidx * inpolgpusize_ / nostreams_, accumulate_ * vdiflen_ * sizeof(unsigned char), cudaMemcpyHostToDevice, gpustreams_[stream]));
            }

            // UnpackKernel<<<cudablocks_[0], cudathreads_[0], 0, gpustreams_[stream]>>>(dinpol_, dunpacked_, nopols_, sampperthread_, rem_, accumulate_ * vdiflen_, 8 / inbits_);
            UnpackKernelOpt<<<50, 1024, 0, gpustreams_[0]>>>(dinpol_, dunpacked_, accumulate_ * vdiflen_);
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
                //cufftCheckError(cufftExecR2C(fftplans_[stream], hdunpacked_[ipol] + stream * unpackedsize_ / nostreams_, hdfft_[ipol] + stream * fftsize_ / nostreams_));
                cufftCheckError(cufftExecR2C(fftplans_[stream], hdunpacked_[ipol], hdfft_[ipol]));
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
            // PowerScaleKernel<<<cudablocks_[1], cudathreads_[1], 0, gpustreams_[stream]>>>(dfft_, filbuffer_ -> GetFilPointer(), dmeans_, dstdevs_, avgfreq_, avgtime_, filchans_, perblock_, stream * fftsize_ / nostreams_, nogulps_, dedispgulpsamples_, dedispextrasamples_, frametime.framet, perframe);
            PowerScaleKernelOpt<<<25, 512, 0, gpustreams_[stream]>>>(dfft_, filbuffer_ -> GetFilPointer(), nogulps_, dedispgulpsamples_, dedispextrasamples_, frametime.framet);
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
                    PrintSafe(ready - 1, "filterbank buffer ready for pool", poolid_);
                }
                filbuffer_ -> SendToRam(ready, dedispstream_, (gulpssent % 2));
                filbuffer_ -> SendToDisk((gulpssent % 2), filheader, telescope_, config_.outdir);
                gulpssent++;
            } else {    // the first buffer will be used for getting the scaling factors
                filbuffer_ -> GetScaling(ready, dedispstream_, dmeans_, dstdevs_);
                scaled_ = true;
                ready = 0;
                if (verbose_) {
                    PrintSafe("Scaling factors have been obtained for pool", poolid_);
                }
            }
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
        numbytes = recvfrom(sockfiledesc_[0], recbufs_[0], vdiflen_ + headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen);
        starttime_.startepoch = (int)(tempbuf[7] & 0x3f);
        starttime_.startsecond = (int)(tempbuf[0] | (tempbuf[1] << 8) | (tempbuf[2] << 16) | ((tempbuf[3] & 0x3f) << 24));
        telescope_ = string() + (char)tempbuf[13] + (char)tempbuf[12];
        cout << starttime_.startepoch << " " << starttime_.startsecond << endl;
    }

    // NOTE: Waiting for frame 0 to start the recording
    while (working_) {
        // NOTE: This will wait forever if nothing is sent to the given port - recvfrom is a blocking call
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], vdiflen_ * headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
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
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], vdiflen_ + headlen_, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            PrintSafe("Error", errno, "on recvfrom on port", recport, "on pool", poolid_);
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
        // TODO: Implement a consumer-producer model, similar to the PAF solution
	    std::copy(recbufs_[portid] + headlen_, recbufs_[portid] + headlen_ + vdiflen_, inpol_[threadid] + vdiflen_ * bufidx);
        if ((threadid == 1) && (frameno > 10))
            readybufidx_[bufidx] = true;

    }
}
