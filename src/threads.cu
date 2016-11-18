#include <cstdlib>
#include <iostream>
#include <queue>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include "buffer.cuh"
#include "config.hpp"
#include "dedisp/dedisp.hpp"
#include "dedisp/DedispPlan.hpp"
#include "errors.hpp"
#include "network.hpp"
#include "pdif.hpp"
#include "pool_multi.cuh"

// Heimdall headers - including might be a bit messy
#include <heimdall/params.hpp>
#include <heimdall/pipeline.hpp>

#include <boost/asio.hpp>
#include <errno.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using boost::asio::ip::udp;
using std::cerr;
using std::cout;
using std::endl;
using std::mutex;
using std::queue;
using std::string;
using std::thread;
using std::vector;

int main(int argc, char *argv[])
{
    std::string config_file;
    config_s config;
    default_config(config);

    // too many parameters to load as arguments - use config file
    if (argc >= 2) {
        for (int ii = 0; ii < argc; ii++) {
            if (std::string(argv[ii]) == "--config") {      // configuration file
                ii++;
                config_file = std::string(argv[ii]);
                read_config(config_file, config);
            }
            if (std::string(argv[ii]) == "-c") {      // the number of chunks to process
                ii++;
                config.chunks = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-s") {     // the number of streams to use
                ii++;
                config.streamno = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-b") {     // the number of telescopes to accept the data from
                ii++;
                config.beamno = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-t") {     // the number of time sample to average
                ii++;
                config.timesavg = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-f") {     // the number of frequency channels to average
                ii++;
                config.freqavg = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-n") {    // the number of GPUs to use
                ii++;
                config.ngpus = atoi(argv[ii]);
            } else if (std::string(argv[ii]) == "-o") {    // output directory for the filterbank files
                ii++;
                struct stat chkdir;
                if (stat(argv[ii], &chkdir) == -1) {
                    cerr << "Stat error" << endl;
                } else {
                    bool isdir = S_ISDIR(chkdir.st_mode);
                    if (isdir)
                        config.outdir = std::string(argv[ii]);
                    else
                        cout << "Output directory does not exist! Will use default directory!";
                }
            } else if (std::string(argv[ii]) == "--gpuid") {
                for (int jj = 0; jj < config.ngpus; jj++) {
                    ii++;
                    config.gpuids.push_back(atoi(argv[ii]));
                }
            } else if (std::string(argv[ii]) == "--ip") {
                for (int jj = 0; jj < config.ngpus; jj++) {
                    ii++;
                    config.ips.push_back(std::string(argv[ii]));
                }
            } else if (std::string(argv[ii]) == "-v") {
                config.verbose = true;
            } else if ((std::string(argv[ii]) == "-h") || (std::string(argv[ii]) == "--help")) {
                cout << "Options:\n"
                        << "\t -h --help - print out this message\n"
                        << "\t --config <file name> - configuration file\n"
                        << "\t - THE USE OF FOLLOWING OPTIONS WILL BE PHASED OUT! USE CONFIG FILE INSTEAD!\n"
                        << "\t -b - the number of beams to process\n"
                        << "\t -c - the number of chunks to process\n"
                        << "\t -f - the number of frequency channels to average\n"
                        << "\t -n - the number of GPUs to use\n"
                        << "\t -o <directory> - output directory\n"
                        << "\t -s - the number of CUDA streams per GPU to use\n"
                        << "\t -t - the number of time samples to average\n"
                        << "\t -v - use verbose mode\n"
                        << "\t --gpuid - GPU IDs to use - the number must be the same as 'n'\n"
                        << "\t --ip - IPs to listen to - the number must be the same as 'n'\n\n";
                exit(EXIT_SUCCESS);
            }
        }

    }

    if (config.verbose) {
        cout << "Starting up. This may take few seconds..." << endl;

        cout << "This is the configuration used:" << endl;
        cout << "\t - gulp size: " << config.gulp << endl;
        cout << "\t - the number of GPUs to use: " << config.ngpus << endl;
        int devcount{0};
        cudaCheckError(cudaGetDeviceCount(&devcount));
        if (config.ngpus > devcount) {
            cout << "You can't use more GPUs than you have available!" << endl;
            config.ngpus = devcount;
        }
        cout << "\t - the number of worker streams per GPU: " << config.streamno << endl;
        cout << "\t - the IP addresses to listen on:" << endl;
        for (int ii = 0; ii < config.ips.size(); ii++) {
            cout << "\t\t * " << config.ips[ii] << endl;
        }
    }

    Oberpool mypool(config);

    std::this_thread::sleep_for(std::chrono::seconds(2));

    cudaDeviceReset();

    return 0;
}
