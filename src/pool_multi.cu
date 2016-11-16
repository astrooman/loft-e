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
#include <numa.h>
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
#include "paf_metadata.hpp"
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

using std::cout;
using std::endl;
using std::mutex;
using std::pair;
using std::queue;
using std::thread;
using std::unique_ptr;
using std::vector;

#define BYTES_PER_WORD 8
#define HEADER 64
#define WORDS_PER_PACKET 896
#define BUFLEN 
#define PORTS 8

mutex cout_guard;

/* ########################################################
TODO: Too many copies - could I use move in certain places?
#########################################################*/

/*####################################################
IMPORTANT: from what I seen in the system files:
There is only one NUMA note.
6 (sic!) physical core, 12 if we really want to use HT
####################################################*/

Oberpool::Oberpool(config_s config) : ngpus(config.ngpus)
{

    for (int ii = 0; ii < ngpus; ii++) {
        gpuvector.push_back(unique_ptr<GPUpool>(new GPUpool(ii, config)));
    }

    for (int ii = 0; ii < ngpus; ii++) {
        threadvector.push_back(thread(&GPUpool::execute, std::move(gpuvector[ii])));
    }

}

Oberpool::~Oberpool(void)
{
    for (int ii = 0; ii < ngpus; ii++) {
        threadvector[ii].join();
    }

}

bool GPUpool::working_ = true;

GPUpool::GPUpool(int id, config_s config) : accumulate(config.accumulate),
                                        beamno{0},
                                        filchans_(config.filchans),
                                        gpuid_(config.gpuids[id]),
                                        strip(config.ips[id]),
                                        highest_buf(0),
                                        batchsize(config.batch),
                                        fftpoint(config.fftsize),
                                        timeavg(config.timesavg),
                                        freqavg(config.freqavg),
                                        nostreams(config.streamno),
                                        npol(config.npol),
                                        poolid_(id),
                                        d_rearrange_size(8 * config.batch * config.fftsize * config.timesavg * config.accumulate),
                                        d_in_size(config.batch * config.fftsize * config.timesavg * config.npol * config.accumulate),
                                        d_fft_size(config.batch * config.fftsize * config.timesavg * config.npol * config.accumulate),
                                        d_time_scrunch_size((config.fftsize - 5) * config.batch * config.accumulate),
                                        d_freq_scrunch_size((config.fftsize - 5) * config.batch  * config.accumulate / config.freqavg),
                                        gulps_sent(0),
                                        gulps_processed(0),
                                        working(true),
                                        scaled_(false)
                                        verbose_(config.verbose)
					                    packcount(0),

{
    avt = min(nostreams + 1, thread::hardware_concurrency());

    config_ = config;

    if (verbose_) {
        cout_guard.lock();
        cout << "Starting GPU pool " << gpuid << endl;
	    cout.flush();
        cout_guard.unlock();
    }
}

void GPUpool::execute(void)
{

    signal(SIGINT, GPUpool::HandleSignal);
    cudaCheckError(cudaSetDevice(gpuid_));

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 3, &cpuset);
    int retaff = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for the GPU pool " << gpuid << endl;
        cout_guard.unlock();
    }

    if(verbose_) {
        cout_guard.lock();
        cout << "GPU pool for device " << gpuid_ << " running on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
    }

    p_dedisp = unique_ptr<DedispPlan>(new DedispPlan(config_.filchans, config_.tsamp, config_.ftop, config_.foff, gpuid));
    p_mainbuffer = unique_ptr<Buffer<float>>(new Buffer<float>(gpuid));

    frame_times = new int[accumulate * nostreams];
    // every thread will be associated with its own CUDA stream
    mystreams = new cudaStream_t[avt];
    // each worker stream will have its own cuFFT plan
    myplans = new cufftHandle[nostreams];

    int nkernels = 4;
    CUDAthreads = new unsigned int[nkernels];
    CUDAblocks = new unsigned int[nkernels];
    nchans = config_.nchans;

    CUDAthreads[0] = 7;
    CUDAthreads[1] = fftpoint * timeavg * batchsize / 42;
    CUDAthreads[2] = nchans;
    CUDAthreads[3] = filchans_;

    CUDAblocks[0] = 48;
    CUDAblocks[1] = 42;
    CUDAblocks[2] = 1;
    CUDAblocks[3] = 1;

    // STAGE: PREPARE THE READ AND FILTERBANK BUFFERS
    if (verbose_)
        cout << "Preparing the memory..." << endl;

    cdesc = cudaCreateChannelDesc<int2>();
    cudaCheckError(cudaPeekAtLastError());

    d_array2Dp = new cudaArray*[nostreams];
    texObj = new cudaTextureObject_t[nostreams];
    rdesc = new cudaResourceDesc[nostreams];
    tdesc = new cudaTextureDesc[nostreams];
    for (int ii = 0; ii < nostreams; ii++) {
        cudaCheckError(cudaMallocArray(&(d_array2Dp[ii]), &cdesc, 7, (batchsize  / 7) * fftpoint * timeavg * accumulate));

        memset(&(rdesc[ii]), 0, sizeof(cudaResourceDesc));
        rdesc[ii].resType = cudaResourceTypeArray;
        rdesc[ii].res.array.array = d_array2Dp[ii];

        memset(&(tdesc[ii]), 0, sizeof(cudaTextureDesc));
        tdesc[ii].addressMode[0] = cudaAddressModeClamp;
        tdesc[ii].filterMode = cudaFilterModePoint;
        tdesc[ii].readMode = cudaReadModeElementType;

        texObj[ii] = 0;
        cudaCheckError(cudaCreateTextureObject(&(texObj[ii]), &(rdesc[ii]), &(tdesc[ii]), NULL));
    }

    // it has to be an array and I can't do anything about that
    sizes[0] = (int)fftpoint;

    // each stream will have its own incoming buffeer to read from
    pack_per_buf = batchsize / 7 * accumulate * nostreams;
    h_pol = new unsigned char[d_rearrange_size * nostreams];
    bufidx_array = new bool[pack_per_buf];
    cudaCheckError(cudaHostAlloc((void**)&h_in, d_rearrange_size * nostreams * sizeof(unsigned char), cudaHostAllocDefault));
    cudaCheckError(cudaMalloc((void**)&d_in, d_in_size * nostreams * sizeof(cufftComplex)));
    cudaCheckError(cudaMalloc((void**)&d_fft, d_fft_size * nostreams * sizeof(cufftComplex)));
    // need to store all 4 Stoke parameters
    dv_time_scrunch.resize(nostreams);
    dv_freq_scrunch.resize(nostreams);
    // TODO: make a private const data memmber and put in the initializer list!!
    stokes = config_.stokes;
    for (int ii = 0; ii < nostreams; ii++) {
        dv_time_scrunch[ii].resize(d_time_scrunch_size * stokes);
        dv_freq_scrunch[ii].resize(d_freq_scrunch_size * stokes);
    }
    // scaling factors memory
    h_means_ = new float*[stokes];
    h_stdevs_ = new float*[stokes];

    // need to be careful what to fill the starting values with
    // we want to have the original data after the scaling in the fist run
    // so we can actually obtain the first scaling factors


    for (int ii = 0; ii < stokes; ii++) {
        cudaCheckError(cudaMalloc((void**)&h_means_[ii], filchans_ * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&h_stdevs_[ii], filchans_ * sizeof(float)));
    }

    cudaCheckError(cudaMalloc((void**)&d_means_, stokes * sizeof(float*)));
    cudaCheckError(cudaMalloc((void**)&d_rstdevs_, stokes * sizeof(float*)));
    cudaCheckError(cudaMemcpy(d_means_, h_means_, 4 * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_rstdevs_, h_stdevs_, 4 * sizeof(float*), cudaMemcpyHostToDevice));

    initscalefactors<<<1,filchans_,0,0>>>(d_means_, d_rstdevs_, stokes);
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaGetLastError());
    // STAGE: PREPARE THE DEDISPERSION
    // generate_dm_list(dm_start, dm_end, width, tol)
    // width is the expected pulse width in microseconds
    // tol is the smearing tolerance factor between two DM trials
    p_dedisp->generate_dm_list(config_.dstart, config_.dend, 64.0f, 1.10f);
    // this is the number of time sample - each timesample will have config.filchans frequencies
    dedisp_totsamples = (size_t)config_.gulp + 1; //p_dedisp->get_max_delay();
    dedisp_buffno = (dedisp_totsamples - 1) / config_.gulp + 1;
    dedisp_buffsize = dedisp_buffno * config_.gulp + 1; //p_dedisp->get_max_delay();
    if (verbose_)
        cout << "Total buffer size: " << dedisp_buffsize << endl;
    // can this method be simplified?
    p_mainbuffer->allocate(accumulate, dedisp_buffno, 1, config_.gulp, dedisp_buffsize, config_.filchans, stokes);
    buffer_ready[0] = false;
    buffer_ready[1] = false;
    p_dedisp->set_killmask(&config_.killmask[0]);
    // everything should be ready for dedispersion after this point

    // STAGE: PREPARE THE SINGLE PULSE SEARCH
    if (verbose_)
        cout << "Setting up dedispersion and single pulse search..." << endl;
    set_search_params(params, config_);
    //commented out for the filterbank dump mode
    //hd_create_pipeline(&pipeline, params);
    // everything should be ready for single pulse search after this point

    // STAGE: start processing
    // FFT threads
    for (int ii = 0; ii < nostreams; ii++) {
            cudaCheckError(cudaStreamCreate(&mystreams[ii]));
            // TODO: add separate error checking for cufft functions
            cufftCheckError(cufftPlanMany(&myplans[ii], 1, sizes, NULL, 1, fftpoint, NULL, 1, fftpoint, CUFFT_C2C, batchsize * timeavg * npol * accumulate));
            cufftCheckError(cufftSetStream(myplans[ii], mystreams[ii]));
            mythreads.push_back(thread(&GPUpool::worker, this, ii));
    }

    // dedispersion thread
    cudaCheckError(cudaStreamCreate(&mystreams[avt - 2]));
    mythreads.push_back(thread(&GPUpool::dedisp_thread, this, avt - 2));

    // STAGE: networking
    if (verbose_)
        cout << "Setting up networking..." << endl;

    memset(&start_time, 0, sizeof(start_time)) ;
    int netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    sfds = new int[PORTS];

    rec_bufs = new unsigned char*[PORTS];
    for (int ii = 0; ii < PORTS; ii++)
        rec_bufs[ii] = new unsigned char[BUFLEN];

    std::ostringstream oss;
    std::string strport;

    // all the magic happens here
    for (int ii = 0; ii < PORTS; ii++) {
        oss.str("");
        oss << 17100 + ii;
        strport = oss.str();

        if((netrv = getaddrinfo(strip.c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
            cout_guard.lock();
            cerr <<  "getaddrinfo() error: " << gai_strerror(netrv) << endl;
            cout_guard.unlock();
        }

        for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
            if((sfds[ii] = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
                cout_guard.lock();
                cerr << "Socket error\n";
                cout_guard.unlock();
                continue;
            }

            if(bind(sfds[ii], tryme->ai_addr, tryme->ai_addrlen) == -1) {
                close(sfds[ii]);
                cout_guard.lock();
                cerr << "Bind error\n";
                cout_guard.unlock();
                continue;
            }
            break;
        }

        if (tryme == NULL) {
            cout_guard.lock();
            cerr << "Failed to bind to the socket " << 17100 + ii << "\n";
            cout_guard.unlock();
        }
    }

    int bufres{4*1024*1024};    // 4MB

    for (int ii = 0; ii < PORTS; ii++) {
        if(setsockopt(sfds[ii], SOL_SOCKET, SO_RCVBUF, (char *)&bufres, sizeof(bufres)) != 0) {
            cout_guard.lock();
            cerr << "Setsockopt error on port " << 17100 + ii << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
    }

    for (int ii = 0; ii < PORTS; ii++)
        receive_threads.push_back(thread(&GPUpool::receive_thread, this, ii));

//    for (int ii = 0; ii < PORTS; ii++)
//        receive_threads[ii].join();

    // TODO: this thread does nothing at this point so might as well make it listen to metadata
    if (verbose_)
        cout << "Setting up metadata logger..." << endl;

    int metabytes, sock_meta;
    addrinfo hints_meta, *servinfo_meta, *tryme_meta;
    memset(&hints_meta, 0, sizeof(hints_meta));
    hints_meta.ai_family = AF_INET;
    hints_meta.ai_socktype = SOCK_DGRAM;
    hints_meta.ai_flags = AI_PASSIVE;
    sockaddr_storage meta_addr;
    memset(&meta_addr, 0, sizeof(meta_addr));
    socklen_t meta_len;

    if ((netrv = getaddrinfo(NULL, "26666", &hints_meta, &servinfo_meta)) != 0) {
        cout_guard.lock();
        cerr << "gettaddrinfo() error on metadata socket 26666" << endl;
        cout_guard.unlock();
    }

        for (tryme_meta = servinfo_meta; tryme_meta != NULL; tryme_meta=tryme_meta->ai_next) {
            if ((sock_meta = socket(tryme_meta->ai_family, tryme_meta->ai_socktype, tryme_meta->ai_protocol)) == -1) {
                cout_guard.lock();
                cerr << "Metadata socket error\n";
                cout_guard.unlock();
                continue;
            }
            if (bind(sock_meta, tryme_meta->ai_addr, tryme_meta->ai_addrlen) == -1) {
                cout_guard.lock();
                cerr << "Metadata bind error\n";
                cout_guard.unlock();
                continue;
            }
            break;
        }

    if (tryme_meta == NULL) {
        cout_guard.lock();
        cerr << "Failed to bind to the metadata socket\n";
        cout_guard.unlock();
    }

    metadata paf_meta;
    ostringstream ossmeta;
    ossmeta << "metadata_log_" << beamno << ".log";
    string metafile = config_.outdir + "/" + ossmeta.str();
    std::fstream metalog(metafile.c_str(), std::ios_base::out | std::ios_base::trunc);

    char *metabuffer = new char[4096];
    meta_len = sizeof(meta_addr);
    if (metalog) {
        /*while(working_) {
            metabytes = recvfrom(sock_meta, metabuffer, 4096, 0, (struct sockaddr*)&meta_addr, &meta_len);
            if (metabytes != 0) {
                string metastr(metabuffer);
                paf_meta.getMetaData(metastr, 0);
                cout << paf_meta.timestamp << "\t";
                cout << paf_meta.beam_num << "\t";
                cout << paf_meta.beam_ra << "\t";
                cout << paf_meta.beam_dec << "\t";
                cout << paf_meta.target_name << endl;
                cout.flush();

                metalog << paf_meta.timestamp << "\t";
                metalog << paf_meta.beam_num << "\t";
                metalog << paf_meta.beam_ra << "\t";
                metalog << paf_meta.beam_dec << "\t";
                metalog << paf_meta.target_name << endl << endl;
            } else {
                cerr << "Got nothing from metadata" << endl;
            }
        }
       */
        metalog.close();
    } else {
        cout_guard.lock();
        cerr << "Metadata log file error!!" << endl;
        cout_guard.unlock();
    }
    delete [] metabuffer;
}

GPUpool::~GPUpool(void)
{
    // TODO: clear the memory properly
    if (verbose_)
        cout << "Calling destructor" << endl;

    for(int ii = 0; ii < mythreads.size(); ii++)
        mythreads[ii].join();

    for (int ii = 0; ii < PORTS; ii++)
        receive_threads[ii].join();

    //save the scaling factors before quitting
    if (scaled_) {
        string scalename = config_.outdir + "/scale_beam_" + std::to_string(beamno) + ".dat";
        std::fstream scalefile(scalename.c_str(), std::ios_base::out | std::ios_base::trunc);

        if (scalefile) {
            float *means = new float[filchans_];
            float *stdevs = new float[filchans_];
            for (int ii = 0; ii < stokes; ii++) {
                cudaCheckError(cudaMemcpy(means, h_means_[ii], filchans_ * sizeof(float), cudaMemcpyDeviceToHost));
                cudaCheckError(cudaMemcpy(stdevs, h_stdevs_[ii], filchans_ * sizeof(float), cudaMemcpyDeviceToHost));
                for (int jj = 0; jj < filchans_; jj++) {
                    scalefile << means[jj] << " " << stdevs[jj] << endl;
                }
                scalefile << endl << endl;
            }
        }
        scalefile.close();
    }
    // cleaning up the stuff
    for (int ii = 0; ii < nostreams; ii++) {
        cudaCheckError(cudaDestroyTextureObject(texObj[ii]));
        cudaCheckError(cudaFreeArray(d_array2Dp[ii]));
    }

    // need deallocation in the dedisp buffer destructor as well
    p_mainbuffer->deallocate();
    // this stuff is deleted in order it appears in the code
    delete [] frame_times;
    delete [] mystreams;
    delete [] CUDAthreads;
    delete [] CUDAblocks;
    delete [] d_array2Dp;
    delete [] texObj;
    delete [] rdesc;
    delete [] tdesc;
    delete [] h_pol;
    delete [] bufidx_array;
    delete [] sfds;
    for (int ii = 0; ii < PORTS; ii++) {
        delete [] rec_bufs[ii];
    }
    delete [] rec_bufs;

    for (int ii = 0; ii < stokes; ii++) {
        cudaCheckError(cudaFree(h_means_[ii]));
        cudaCheckError(cudaFree(h_stdevs_[ii]));
    }

    delete [] h_means_;
    delete [] h_stdevs_;
    cudaCheckError(cudaFree(d_means_));
    cudaCheckError(cudaFree(d_rstdevs_));


    cudaCheckError(cudaFree(d_in));
    cudaCheckError(cudaFree(d_fft));
    cudaCheckError(cudaFreeHost(h_in));

    for (int ii = 0; ii < nostreams; ii++) {
        cufftCheckError(cufftDestroy(myplans[ii]));
    }

    delete [] myplans;
}

void GPUpool::HandleSignal(int signum) {

    cout << "Captured the signal\nWill now terminate!\n";
    working_ = false;
}

void GPUpool::worker(int stream)
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8 + 1 + (int)(stream / 1), &cpuset);
    int retaff = pthread_setaffinity_np(mythreads[stream].native_handle(), sizeof(cpu_set_t), &cpuset);

    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for stream " << stream << endl;
        cout_guard.unlock();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Starting worker " << gpuid << ":" << stream << " on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
    }

    cudaSetDevice(gpuid);
    unsigned int skip = stream * d_in_size;

    unsigned int current_frame;

    float *pdv_time_scrunch = thrust::raw_pointer_cast(dv_time_scrunch[stream].data());
    float *pdv_freq_scrunch = thrust::raw_pointer_cast(dv_freq_scrunch[stream].data());

    float **p_fil = p_mainbuffer->get_pfil();
    float **pd_fil;
    cudaMalloc((void**)&pd_fil, stokes * sizeof(float *));
    cudaMemcpy(pd_fil, p_fil, stokes * sizeof(float *), cudaMemcpyHostToDevice);

    int skip_read = stream * (pack_per_buf / nostreams);
    int skip_to_end = (stream + 1) * (pack_per_buf / nostreams) - 1;
    int next_start;
    if (stream != 3) {
        next_start = skip_to_end + 24;
    } else {
        next_start = 23;
    }
    bool endready = false;
    bool innext = false;
    while (working_) {
        endready = false;
        innext = false;
        for (int ii = 0; ii < 4; ii++) {
            //cout << bufidx_array[skip_to_end - ii] << " " << bufidx_array[skip_to_end + 24 - ii] << endl;
            endready = endready || bufidx_array[skip_to_end - ii];
            innext = innext || bufidx_array[next_start - ii];
        }
        if (endready && innext) {
            for (int ii = 0; ii < 4; ii++) {
                bufidx_array[skip_to_end - ii] = false;
                bufidx_array[next_start - ii] = false;
            }
            std::copy(h_pol + stream * d_rearrange_size,  h_pol + stream * d_rearrange_size + d_rearrange_size, h_in + stream * d_rearrange_size);;
            for (int frameidx = 0; frameidx < accumulate; frameidx++) {
                if (frame_times[stream * accumulate + frameidx] != 0) {
                    current_frame = frame_times[stream * accumulate + frameidx];
                    break;
                }
            }
            for (int frameidx = 0; frameidx < accumulate; frameidx++)
                frame_times[stream * accumulate + frameidx] = 0;

            obs_time frame_time{start_time.start_epoch, start_time.start_second, current_frame};
            cudaCheckError(cudaMemcpyToArrayAsync(d_array2Dp[stream], 0, 0, h_in + stream * d_rearrange_size, d_rearrange_size, cudaMemcpyHostToDevice, mystreams[stream]));
            cufftCheckError(cufftExecC2C(myplans[stream], d_in + skip, d_fft + skip, CUFFT_FORWARD));
            powertime2<<<48, 27, 0, mystreams[stream]>>>(d_fft + skip, pdv_time_scrunch, d_time_scrunch_size, timeavg, accumulate);
            addchanscale<<<CUDAblocks[3], CUDAthreads[3], 0, mystreams[stream]>>>(pdv_time_scrunch, pd_fil, (short)filchans_, config_.gulp, dedisp_buffsize, dedisp_buffno, d_time_scrunch_size, freqavg, current_frame, accumulate, d_means_, d_rstdevs_);
            cudaStreamSynchronize(mystreams[stream]);
            // used to check for any possible errors in the kernel execution
            cudaCheckError(cudaGetLastError());
            //cout << current_frame << endl;
            //cout.flush();
            p_mainbuffer->update(frame_time);
            //working_ = false;
        } else {
            std::this_thread::yield();
        }
    }

    cudaFree(pd_fil);
}

void GPUpool::dedisp_thread(int dstream)
{

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8, &cpuset);
    int retaff = pthread_setaffinity_np(mythreads[nostreams].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cout_guard.lock();
        cout << "Error setting thread affinity for dedisp thread" << endl;
        cout_guard.unlock();
    }

    cudaCheckError(cudaSetDevice(gpuid));
    if (verbose_)
        cout << "Dedisp thread up and running..." << endl;
    int ready{0};
    while(working_) {
        ready = p_mainbuffer->ready();
        if (ready) {
            if (scaled_) {
                header_f headerfil;
                headerfil.raw_file = "tastytastytest";
                headerfil.source_name = "J1641-45";
                headerfil.az = 0.0;
                headerfil.dec = 0.0;
                headerfil.fch1 = config_.ftop;
                headerfil.foff = config_.foff;
                headerfil.ra = 0.0;
                headerfil.rdm = 0.0;
                headerfil.tsamp = config_.tsamp;
                // TODO: this totally doesn't work when something is skipped
                headerfil.tstart = get_mjd(start_time.start_epoch, start_time.start_second + gulps_sent * config_.gulp * config_.tsamp);
                headerfil.za = 0.0;
                headerfil.data_type = 1;
                headerfil.ibeam = beamno;
                headerfil.machine_id = 2;
                headerfil.nbeams = 1;
                headerfil.nbits = 32;
                headerfil.nchans = config_.filchans;
                headerfil.nifs = 1;
                headerfil.telescope_id = 2;

                if (verbose_)
                    cout << ready - 1 << " buffer ready " << endl;
                p_mainbuffer->send(d_dedisp, ready, mystreams[dstream], (gulps_sent % 2));
                p_mainbuffer->dump((gulps_sent % 2), headerfil, config_.outdir);
                gulps_sent++;
                //working = false;
            } else {
                // perform the scaling
                p_mainbuffer->rescale(ready, mystreams[dstream], d_means_, d_rstdevs_);
                cudaCheckError(cudaGetLastError());
                scaled_ = true;
                ready = 0;
                if (verbose_)
                    cout << "Scaling factors have been obtained" << endl;
            }
        } else {
            std::this_thread::yield();
        }
    }
}

void GPUpool::receive_thread(int ii)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((int)(poolid_) * 8 + 1 + nostreams + (int)(ii / 3), &cpuset);
    int retaff = pthread_setaffinity_np(receive_threads[ii].native_handle(), sizeof(cpu_set_t), &cpuset);
    if (retaff != 0) {
        cout_guard.lock();
        cerr << "Error setting thread affinity for receive thread on port " << 17000 + ii << endl;
        cout_guard.unlock();
    }

    if (verbose_) {
        cout_guard.lock();
        cout << "Receive thread on port " << 17000 + ii << " running on CPU " << sched_getcpu() << endl;
        cout_guard.unlock();
    }

    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));

    const int pack_per_worker_buf = pack_per_buf / nostreams;
    int numbytes{0};
    short bufidx{0};
    // this will always be an integer
    int frame{0};
    int refs{0};
    int packcount{0};
    int group{0};

    if (ii == 0) {
        unsigned char *tempbuf = recbufs[0];
        numbytes = recvfrom(sfds[0], recbufs[0], BUFLEN, 0, (struct sockaddr*)&their_addr, &addr_len);
        start_time.start_epoch = (int)(tempbuf[4] >> 0x3f);
        start_time.start_second = (int)(tempbuf[3] | (temp_buf[2] << 8) | (temp_buf[1] << 16) | ((temp_buf[0] & 0x3f) << 24));
    }

    while (true) {
        if ((numbytes = recvfrom(sfds[ii], recbufs[ii], BUFLEN, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << 17100 + ii << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;
        frame = (int)(recbufs[ii][7] | (recbufs[ii][6] << 8) | (recbufs[ii][5] << 16));
        if (frame == 0) {
            break;
        } // wait until reaching frame zero of next second before beginnning recording
    }

    while(working_) {
        if ((numbytes = recvfrom(sfds[ii], recbufs[ii], BUFLEN, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << 17100 + ii << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;
        refs = (int)(recbufs[ii][3] | (recbufs[ii][2] << 8) | (recbufs[ii][1] << 16) | ((recbufs[ii][0] & 0x3f) << 24));
        frame = (int)(recbufs[ii][7] | (recbufs[ii][6] << 8) | (recbufs[ii][5] << 16);
        frame = frame + (refs - start_time.start_second - 1) / 1 * 4000; //running tally of total frames (1 pol) subtract 1 because you skip the first second as it prob. didn't start recording on frame zero

        // looking at how much stuff we are not missing - remove a lot of checking for now
        // TODO: add some mininal checks later anyway
        bufidx = (((int)frame / accumulate) % nostream) * accumulate) + (frame % accumulate);
        frametimes[bufidx] = frame;
        std::copy(recbufs[ii] + HEADER, recbufs[ii] + BUFLEN, h_pol[polid_] + (BUFLEN - HEADER) * bufidx)//copy data
        bufidxarray[bufidx] = true;
        //}
    }
}
