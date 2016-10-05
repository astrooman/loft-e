#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include <cufft.h>
#include <pthread.h>
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
using std::thread;
using std::queue;
using std::vector;

#define BYTES_PER_WORD 8
#define HEADER 64
#define BUFLEN 7168 + 64
#define PORTS 8 
#define SIZE 1200 
#define WORDS_PER_PACKET 896

struct obs_time {

    int start_epoch;            // reference epoch at the start of the observation
    int start_second;           // seconds from the reference epoch at the start of the observation
    int framet;                 // frame number from the start of the observation

};

void* get_in_addr(struct sockaddr *sadr)
{
    if (sadr->sa_family == AF_INET) {
        return &(((sockaddr_in*)sadr)->sin_addr);
    }
    return &(((sockaddr_in6*)sadr)->sin6_addr);
}


class GPUpool {
    private:
        vector<thread> mythreads;
        vector<thread> receive_threads;
        mutex datamutex;
        mutex packmutex;
        queue<vector<cufftComplex>> mydata;
        size_t highest_buf;
        int highest_frame;
        bool buffer_ready[2];
        unsigned char *h_pol;
        cufftComplex *all_data;
        int d_in_size;
        unsigned int accumulate = 65536;
        short nchans = 336;
        short npol = 2;
        short fftsize = 32;
        short timesavg = 4;
        short pack_per_buf;

        unsigned char **rec_bufs;
        //unsigned char *temp_buf;
        int *sfds;
        int **frames;
        int *packcount;
        int wait;
    protected:

    public:
        GPUpool(void);
        ~GPUpool(void);
    //    void receive_handler(const boost::system::error_code& error, std::size_t bytes_transferred, udp::endpoint endpoint, int ii);
        void receive_thread(int ii);
};

GPUpool::GPUpool(void) : highest_buf(0), highest_frame(-1) {

    cout << "GPUpool constructor..." << endl;

    buffer_ready[0] = false;
    buffer_ready[1] = false;
    d_in_size = nchans * npol * fftsize * timesavg;

    all_data = new cufftComplex[SIZE * 7 * 128 * PORTS];

    int sfd, rv;
    addrinfo hints, *servinfo, *p;
    //sockaddr_storage their_addr;
    unsigned char frame[BUFLEN];

    rec_bufs = new unsigned char*[PORTS];

    cout << "Memallocs on CPU " << sched_getcpu() << endl;

    for (int ii = 0; ii < PORTS; ii++)
        rec_bufs[ii] = new unsigned char[BUFLEN];

    char s[INET6_ADDRSTRLEN];

    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    
    hints.ai_flags = AI_PASSIVE;

    std::ostringstream oss;
    std::string strport;
 
    sfds = new int[PORTS];

    for (int ii = 0; ii < PORTS; ii++) {

        oss.str("");
        oss << 17100 + ii;
        strport = oss.str();

        if((rv = getaddrinfo("10.17.0.1", strport.c_str(), &hints, &servinfo)) != 0) {
            cout << "getaddrinfo error " << gai_strerror(rv) << endl;
            exit(EXIT_FAILURE);
        } 

        for (p = servinfo; p != NULL; p = p->ai_next) {
            if((sfds[ii] = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
                cout << "Socket error\n";
                continue;
            }

            if (bind(sfds[ii], p->ai_addr, p->ai_addrlen) == -1) {
                close(sfds[ii]);
                cout << "Error on bind\n";
                continue;
            }
      
            break;
        }

        if (p == NULL) {
            cout << "Failed to bind the socket\n";
            exit(EXIT_FAILURE);
        }

    }

    packcount = new int[PORTS];
    for (int ii = 0; ii < PORTS; ii++)
        packcount[ii] = 0;
    frames = new int*[PORTS];

    int n = 4 * 1024 * 1024;

    for (int ii = 0; ii < PORTS; ii++) {
        if((setsockopt(sfds[ii], SOL_SOCKET, SO_RCVBUF, (char*)&n, sizeof(n))) == -1) {
	    cout << "Option error for socket " << 17100 + ii << "...\n";
            cout << errno << endl;
        }
    }
    
    for (int ii = 0; ii < PORTS; ii++)
        frames[ii] = new int[SIZE];

    wait = true;

    for (int ii = 0; ii < PORTS; ii++)
        receive_threads.push_back(thread(&GPUpool::receive_thread, this, ii));

    wait = false;

    std::this_thread::sleep_for(std::chrono::seconds(2));

    for (int ii = 0; ii < receive_threads.size(); ii++)
        receive_threads[ii].join();
    cout << "Done receiving..." << endl;
    cout.flush();
}

void GPUpool::receive_thread(int ii)
{
    int portorder = ii;
    cout << "Starting to listen on port " << 17100 + ii << "..." << endl;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(9 + (int)(ii / 3),  &cpuset);

    pthread_t current_thread = pthread_self();
    if(pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
        cout << "Problems with setting the affinity!!" << endl;
    }

    unsigned char* temp_buf = new unsigned char[BUFLEN];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    char huh[INET6_ADDRSTRLEN];
    int numbytes{0};
    int fpga{0};
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));
    while(wait) {}
    if (ii == 0) {

        if (( numbytes = recvfrom(sfds[ii], rec_bufs[0], BUFLEN-1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
                cout << "Error on initial recvfrom\n";
                cout << errno << endl;
        }
        unsigned char *temp_buf = rec_bufs[0];
        cout << "Start frame: " << (int)(temp_buf[7] | (temp_buf[6] << 8) | (temp_buf[5] << 16) | (temp_buf[4] << 24)) << endl;
        cout << "Start reference epoch: " << (int)(temp_buf[12] >> 2) << endl;
        cout << "Start second from the reference epoch: " << (int)(temp_buf[3] | (temp_buf[2] << 8) | (temp_buf[1] << 16) | ((temp_buf[0] & 0x3f) << 24)) << endl;
    }
    
    int frame{0};

    while (true) {
        if ((numbytes = recvfrom(sfds[ii], rec_bufs[ii], BUFLEN - 1, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
            cout << "Error of recvfrom on port " << 17100 + ii << endl;
            // possible race condition here
            cout << "Errno " << errno << endl;
        }
        if (numbytes == 0)
            continue;
        frame = (int)(rec_bufs[ii][7] | (rec_bufs[ii][6] << 8) | (rec_bufs[ii][5] << 16) | (rec_bufs[ii][4] << 24));
        if (frame == 0) {
            break;
        }
    }

    cufftComplex *port_data = new cufftComplex[SIZE * 7 * 128];
    int idx{0};
    unsigned char *data = new unsigned char[BUFLEN-HEADER];
    while(packcount[portorder] < SIZE) {
        if (( numbytes = recvfrom(sfds[ii], rec_bufs[ii], BUFLEN, 0, (struct sockaddr*)&their_addr, &addr_len)) == -1) {
                cout << "Error on recvfrom\n";
                cout << errno << endl;
        }
        if (numbytes == 0)
            continue;
        frame = (int)(rec_bufs[ii][7] | (rec_bufs[ii][6] << 8) | (rec_bufs[ii][5] << 16) | (rec_bufs[ii][4] << 24));
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2;

        std::copy(rec_bufs[ii] + HEADER, rec_bufs[ii] + BUFLEN, data);

        for (int sample = 0; sample < 128 * 7; sample++) {
                idx = 8 * sample;
                port_data[packcount[portorder] * 7 * 128 + sample].x = static_cast<float>(static_cast<short>(data[HEADER + idx + 7] | (data[HEADER + idx + 6] << 8)));
                port_data[packcount[portorder] * 7 * 128 + sample].y = static_cast<float>(static_cast<short>(data[HEADER + idx + 5] | (data[HEADER + idx + 4] << 8)));
        }
 
        packcount[portorder]++;
    }
    
    std::copy(port_data, port_data + SIZE * 7 * 128, all_data + SIZE * 7 * 128 * portorder);
}

GPUpool::~GPUpool(void) {

    cout << "GPUpool destructor..." << endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    cout << "Deleted the thing..." << endl;
    cout.flush();
  
    std::ofstream rawdata("raw_data.dat", std::ios_base::trunc | std::ios_base::out);
    if(rawdata) {
        for (int jj = 0; jj < SIZE * PORTS * 128 * 7; jj++)
            rawdata << all_data[jj].x << " " << all_data[jj].y << endl;
    }
    rawdata.close();   
}

int main(int argc, char *argv[])
{
    try {

        cout << "Main is on CPU " << sched_getcpu() << endl;

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(8, &cpuset);
        pthread_t this_thread = pthread_self();
        int rval = pthread_setaffinity_np(this_thread, sizeof(cpu_set_t), &cpuset);
        if (rval != 0)
            cout << "Problem with main thread affinity!!!!" << endl;

        cout << "Main is now on CPU " << sched_getcpu() << endl;

        GPUpool mypool;

    } catch (std::exception &e) {
        cout << "Something bad happened: " << e.what() << endl;
    }

    return 0;
}
