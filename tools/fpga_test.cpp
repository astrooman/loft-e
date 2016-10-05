#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <sstream>
#include <thread>
#include <vector>

#include <inttypes.h>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

#define PORTS 8
#define SIZE 16384

using std::cout;
using std::endl;
using std::mutex;
using std::thread;
using std::vector;

mutex printmutex;

void receive_thread(int ii, std::string strip) {

    printmutex.lock();
    cout << "Starting receive on port " << 17100 + ii << endl;
    printmutex.unlock();

    int sfd, netrv;
    addrinfo hints, *servinfo, *tryme;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;
  
    std::ostringstream oss;
    std::string strport;
    oss.str("");
    oss << 17100 + ii;
    strport = oss.str();
    

    if((netrv = getaddrinfo(strip.c_str(), strport.c_str(), &hints, &servinfo)) != 0) {
        cout << "getaddrinfo() error: " << gai_strerror(netrv) << endl;
    }

    for (tryme = servinfo; tryme != NULL; tryme=tryme->ai_next) {
        if((sfd = socket(tryme->ai_family, tryme->ai_socktype, tryme->ai_protocol)) == -1) {
            cout << "Socket error\n";
            continue;
        }

        if(bind(sfd, tryme->ai_addr, tryme->ai_addrlen) == -1) {
            close(sfd);
            cout << "Bind error\n";
            continue;
        }
        break;
    }

    if (tryme == NULL) {
        cout << "Failed to bind to the socket\n";
    }

    int fpga, numbytes;
    int *fpgaid = new int[6];
    unsigned char *rec_buf = new unsigned char[7168 + 64];
    sockaddr_storage their_addr;
    memset(&their_addr, 0, sizeof(their_addr));
    socklen_t addr_len;
    memset(&addr_len, 0, sizeof(addr_len));
    
    for (int jj = 0; jj < SIZE; jj++) {
        numbytes = recvfrom(sfd, rec_buf, 7168 + 64, 0, (struct sockaddr*)&their_addr, &addr_len);
        fpga = ((short)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 16) & 0xff) - 1) * 6 + ((int)((((struct sockaddr_in*)&their_addr)->sin_addr.s_addr >> 24)& 0xff) - 1) / 2; 
        fpgaid[fpga % 6]++; 
    }

    printmutex.lock();
    cout << "Received on port " << 17100 + ii << endl;
    for (int jj = 0; jj < 6; jj++) {
        cout << "FPGA " << ii * 6 + jj << ": " << fpgaid[jj] << endl;
    }
    cout << endl;
    printmutex.unlock();


}

int main(int argc, char *argv[])
{

    vector<thread> mythreads;    

    printmutex.lock();
    cout << "Recording data on " << argv[1] << endl;
    printmutex.unlock();

    for (int ii = 0; ii < PORTS; ii++) {
        mythreads.push_back(thread(receive_thread, ii, std::string(argv[1])));
    }

    for (int ii = 0; ii < PORTS; ii++) {
        mythreads[ii].join();
    }

    return 0;
}

