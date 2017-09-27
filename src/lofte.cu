#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sys/stat.h>
#include <vector>

#include <cuda.h>
#include <cufft.h>

#include "config.hpp"
#include "errors.hpp"
#include "main_pool.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::string;
using std::vector;

int main(int argc, char *argv[])
{
    std::string configfile;
    InConfig config;
    SetDefaultConfig(config);

    if (argc >= 2) {
        for (int iarg = 0; iarg < argc; iarg++) {
            if (std::string(argv[iarg]) == "--config") {      // configuration file
                iarg++;
                configfile = std::string(argv[iarg]);
                try {
                    ReadConfig(configfile, config);
                } catch (const exception &exc) {
                    cout << exc.what() << endl;
                    return 1;
                }
            }
            if (std::string(argv[iarg]) == "-s") {     // the number of streams to use
                iarg++;
                config.nostreams = atoi(argv[iarg]);
            } else if (std::string(argv[iarg]) == "-b") {     // the number of telescopes to accept the data from
                iarg++;
                config.nobeams = atoi(argv[iarg]);
            } else if (std::string(argv[iarg]) == "-t") {     // the number of time sample to average
                iarg++;
                config.timeavg = atoi(argv[iarg]);
            } else if (std::string(argv[iarg]) == "-f") {     // the number of frequency channels to average
                iarg++;
                config.freqavg = atoi(argv[iarg]);
            } else if (std::string(argv[iarg]) == "-n") {    // the number of GPUs to use
                iarg++;
                config.nogpus = atoi(argv[iarg]);
            } else if (std::string(argv[iarg]) == "-o") {    // output directory for the filterbank files
                iarg++;
                struct stat chkdir;
                if (stat(argv[iarg], &chkdir) == -1) {
                    cerr << "Stat error" << endl;
                } else {
                    bool isdir = S_ISDIR(chkdir.st_mode);
                    if (isdir)
                        config.outdir = std::string(argv[iarg]);
                    else
                        cout << "Output directory does not exist! Will use default directory!";
                }
            } else if (std::string(argv[iarg]) == "--gpuid") {
                for (int igpu = 0; igpu < config.nogpus; igpu++) {
                    iarg++;
                    config.gpuids.push_back(atoi(argv[iarg]));
                }
            } else if (std::string(argv[iarg]) == "--ip") {
                for (int iip = 0; iip < config.nogpus; iip++) {
                    iarg++;
                    config.ips.push_back(std::string(argv[iarg]));
                }
            } else if (std::string(argv[iarg]) == "-v") {
                config.verbose = true;
            } else if ((std::string(argv[iarg]) == "-h") || (std::string(argv[iarg]) == "--help")) {
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
        PrintConfig(config);
    }

    MainPool workpool(config);

    cudaDeviceReset();

    return 0;
}
