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

#define BYTES_PER_WORD 8
#define HEADER 64
#define WORDS_PER_PACKET 896
#define BUFLEN 8000
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

Oberpool::Oberpool(InConfig config) : ngpus(config.ngpus)
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

GPUpool::GPUpool(int id, InConfig config) : accumulate_(config.accumulate),
                                            gpuid_(config.gpuids[id]),
                                            nostreams_(config.streamno),
                                            poolid_(id),
                                            ports_(config.ports[id]),
                                            // strip_(config.ips[id]),
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

void GPUpool::execute(void)
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
        recbufs_[iport] = new unsigned char[BUFLEN];

    ostringstream ssport;
    string strport;

    // all the magic happens here
    for (int iport = 0; iport < noports_; iport++) {
        ssport.str("");
        ssport << ports_[iport];
        strport = ssport.str();

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

GPUpool::~GPUpool(void)
{

}

void GPUpool::HandleSignal(int signum) {

    cout << "Captured the signal\nWill now terminate!\n";
    working_ = false;
}

void GPUpool::ReceiveData(int portid, int recport)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // TODO: need to test how much we can squeeze out of the single core
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
    int frameno{0};
    int refsecond{0};
    int threadid{0};
    int packcount{0};

    // TODO: be careful which port waits
    if (recport == ports_[0]) {
        unsigned char *tempbuf = recbufs_[0];
        numbytes = recvfrom(sockfiledesc_[0], recbufs_[0], BUFLEN, 0, (struct sockaddr*)&theiraddr, &addrlen);
        starttime_.startepoch = (int)(tempbuf[4] & 0x3f);
        starttime_.startsecond = (int)(tempbuf[3] | (tempbuf[2] << 8) | (tempbuf[1] << 16) | ((tempbuf[0] & 0x3f) << 24));
    }

    while (true) {
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], BUFLEN, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << recport << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;
        frameno = (int)(recbufs_[portid][7] | (recbufs_[portid][6] << 8) | (recbufs_[portid][5] << 16));
        if (frameno == 0) {
            break;
        } // wait until reaching frame zero of next second before beginnning recording
    }

    while(working_) {
        if ((numbytes = recvfrom(sockfiledesc_[portid], recbufs_[portid], BUFLEN, 0, (struct sockaddr*)&theiraddr, &addrlen)) == -1) {
            cout_guard.lock();
            cerr << "Error of recvfrom on port " << recport << endl;
            cerr << "Errno " << errno << endl;
            cout_guard.unlock();
        }
        if (numbytes == 0)
            continue;
        refsecond = (int)(recbufs_[portid][3] | (recbufs_[portid][2] << 8) | (recbufs_[portid][1] << 16) | ((recbufs_[portid][0] & 0x3f) << 24));
        frameno = (int)(recbufs_[portid][7] | (recbufs_[portid][6] << 8) | (recbufs_[portid][5] << 16));
        frameno = frameno + (refsecond - starttime_.startsecond - 1) / 1 * 4000; //running tally of total frames (1 pol) subtract 1 because you skip the first second as it prob. didn't start recording on frame zero

        // looking at how much stuff we are not missing - remove a lot of checking for now
        // TODO: add some mininal checks later anyway
        bufidx = (((int)frameno / accumulate_) % nostreams_) * accumulate_ + (frameno % accumulate_);
        // frametimes[bufidx] = frameno;
        //std::copy(recbufs[portid] + HEADER, recbufs_[portid] + BUFLEN, h_pol_[polid_] + (BUFLEN - HEADER) * bufidx)//copy data
        //bufidxarray[bufidx] = true;
        //}
    }
}
