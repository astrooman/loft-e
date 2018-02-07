#ifndef _H_LOFTE_CONFIG
#define _H_LOFTE_CONFIG

#include <algorithm>
#include <chrono>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <heimdall/params.hpp>

struct InConfig {
    bool combine;
    bool test;
    bool verbose;

    double band;                // sampling rate for each band in MHz
    double dmend;
    double dmstart;
    double foff;                // channel width in MHz
    double ftop;                // frequency of the top channel in MHz
    double tsamp;               // sampling time

    std::chrono::system_clock::time_point recordstart;

    std::string outdir;

    std::string ip;
    std::vector<int> killmask;
    std::vector<int> ports;


    unsigned int accumulate;        //!< MOVED TO DEFINE
    unsigned int batch;             //!< NOT IN USE
    unsigned int fftsize;           //!< Single FFT size
    unsigned int filchans;          //!< Number of output filterbank channels
    unsigned int freqavg;           //!< Number of frequency channels to average
    unsigned int gpuid;             //!< Id of the GPU to use
    unsigned int gulp;              //!< Dedispersion gulp size
    unsigned int half;              //!< Which half of the CPU to use
    unsigned int headlen;           //!< MOVED TO DEFINE
    unsigned int inbits;            //!< MOVED TO DEFINE
    unsigned int nobeams;           //!< Number of beams per node
    unsigned int nochans;           //!< number of 1MHz channels - REMOVE
    unsigned int nogpus;            //!< number of GPUs to use - REMOVE
    unsigned int nopols;            //!< Number of incoming polarisations
    unsigned int noports;           //!< Number of ports used for receiving
    unsigned int nostokes;          //!< Number of Stokes parameters to compute
    unsigned int nostreams;         //!< Number of GPU streams to use for filterbank
    unsigned int record;            //!< Number of seconds to record
    unsigned int vdiflen;           //!< MOVED TO DEFINE
    unsigned int scaleseconds;      //!< Number of seconds to use for the incoming data
    unsigned int timeavg;           //!< Number of time samples to average

};

inline void SetDefaultConfig(InConfig &config) {
    config.test = false;
    config.verbose = false;

    config.band = 64;
    config.dmend = 4000.0;
    config.dmstart = 0.0;
    config.ftop = 1400.0;

    config.accumulate = 4000;
    config.fftsize = 256;
    config.freqavg = 1;
    config.gulp = 131072;     // 2^17, equivalent to ~14s for 108us sampling time
    config.headlen = 32;
    config.inbits = 2;
    config.nobeams = 1;
    config.nogpus = 1;
    config.nopols = 2;
    config.nostokes = 1;
    config.nostreams = 1;
    config.outdir = "./";
    config.record = 300;            //!< Record 5 minutes of data
    config.scaleseconds = 5;
    config.timeavg = 16;
    config.vdiflen = 8000;

    config.batch = config.nochans;
    config.foff = config.band / config.fftsize * (double)config.freqavg;
    config.tsamp = (double)1.0 / (2.0 * config.band * 1e+06) * config.fftsize * 2.0 * (double)config.timeavg;
    for (int ichan = 0; ichan < config.filchans; ichan++)
         (config.killmask).push_back((int)1);
}

inline void PrintConfig(const InConfig &config) {

    std::cout << "Configuration overview: " << std::endl;
    std::cout << "\t - the number of GPUs to use: " << config.nogpus << std::endl;
    std::cout << "\t - IP to listen on: " << config.ip << std::endl;
    std::cout << "\t - ports to listen on: " << std::endl;
    for (auto port : config.ports) {
        std::cout << "\t\t * " << port << std::endl;
    }
    std::cout << "\t - output directory: " << config.outdir << std::endl;
    time_t tmptime = std::chrono::system_clock::to_time_t(config.recordstart);
    std::cout << "\t - recording start time (experimental): " << std::asctime(std::gmtime(&tmptime)) << std::endl;
    std::cout << "\t - the number of seconds to record: " << config.record << std::endl;
    std::cout << "\t - the number of seconds to use for scaling factors:" << config.scaleseconds << std::endl;
    std::cout << "\t - number of channels to average: " << config.freqavg << std::endl;
    std::cout << "\t - number of time samples to average:" << config.timeavg << std::endl;
    std::cout << "\t - dedisperse gulp size: " << config.gulp << std::endl;
    std::cout << "\t - start DM: " << config.dmstart << std::endl;
    std::cout << "\t - end DM: " << config.dmend << std::endl;

}

inline void ReadConfig(std::string filename, InConfig &config) {

    std::fstream inconfig(filename.c_str(), std::ios_base::in);
    std::string line;
    std::string paraname;
    std::string paravalue;

    if(inconfig) {
        while(std::getline(inconfig, line)) {
            std::istringstream ossline(line);
            ossline >> paraname >> paravalue;

            if (paraname == "DMEND") {
                config.dmend = stod(paravalue);
            } else if (paraname == "DMSTART") {
                config.dmstart = stod(paravalue);
            } else if (paraname == "FCENT") {
                std::istringstream ssvalue(paravalue);
                std::string bandcentre;
                while(std::getline(ssvalue, bandcentre, ',')) {
                    config.centres.push_back(std::stod(bandcentre)));
                }
            }
            } else if (paraname == "FFTSIZE") {
                config.fftsize = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "FOFF") {
                std::istringstream ssvalue(paravalue);
                std::string bandoff;
                while(std::getline(ssvalue, bandoff, ',')) {
                    config.bands.push_back(std::stod(bandoff));
                }
            } else if (paraname == "FREQAVG") {
                config.freqavg = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "DEDISPGULP") {
                config.gulp = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "GPUID") {
                config.gpuid = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "IP") {
                config.ip = paravalue;
            } else if (paraname == "NO1MHZCHANS") {
                config.nochans = (unsigned int)(std::stoi(paravalue));
                config.batch = config.nochans;
            } else if (paraname == "NOBEAMS") {
                config.nobeams = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "NOGPUS") {
                config.nogpus = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "NOPOLS") {
                config.nopols = std::stoi(paravalue);
            } else if (paraname == "NOSTOKES") {
                config.nostokes = std::stoi(paravalue);
            } else if (paraname == "NOSTREAMS") {
                config.nostreams = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "OUTDIR") {
                struct stat chkdir;
                if (stat(paravalue.c_str(), &chkdir) == -1) {
                    std::cerr << "Stat error" << std::endl;
                } else {
                    bool isdir = S_ISDIR(chkdir.st_mode);
                    if (isdir)
                        config.outdir = paravalue;
                    else
                        std::cout << "Output directory does not exist! Will use the default directory!" << std::endl;
                }
            } else if (paraname == "PORTS") {
                std::istringstream ssvalue(paravalue);
                std::string ipports;
                while(std::getline(ssvalue, ipports, ';')) {
                    std::vector<int> vtmp;
                    std::istringstream ssports(ipports);
                    std::string singleport;
                    while(std::getline(ssports, singleport, ',')) {
                        config.ports.push_back(stoi(singleport));
                    }
                }
            } else if (paraname == "RECORD") {
                config.record = std::stod(paravalue);
            } else if (paraname == "SCALE") {
                config.scaleseconds = std::stoi(paravalue);
            } else if (paraname == "TIME_AVERAGE") {
                config.timeavg = (unsigned int)(std::stoi(paravalue));
            } else {
                std::cout << "Error: unrecognised parameter: " << paraname << std::endl;
            }
        }
    } else {
        std::cout << "Error opening the configuration file!!\n Will use default configuration instead." << std::endl;
    }

    config.tsamp = (double)1.0 / (config.band * 1e+06) * config.fftsize * 2 * (double)config.timeavg;
    config.foff = config.band / config.fftsize * (double)config.freqavg;

    inconfig.close();
}

inline void set_search_params(hd_params &params, InConfig config)
{
    params.verbosity       = 0;
    #ifdef HAVE_PSRDADA
    params.dada_id         = 0;
    #endif
    params.sigproc_file    = NULL;
    params.yield_cpu       = false;
    params.nsamps_gulp     = config.gulp;
    // TODO: This is no longer being used
    params.dm_gulp_size    = 2048;//256;    // TODO: Check that this is good
    params.baseline_length = 2.0;
    params.beam            = 0;
    params.override_beam   = false;
    params.nchans          = config.filchans;
    params.dt              = config.tsamp;
    params.f0              = config.ftop;
    params.df              = -abs(config.foff);    // just to make sure it is negative
    // no need for dm params as the code will not do it
    params.dm_min          = config.dmstart;
    params.dm_max          = config.dmend;
    params.dm_tol          = 1.25;
    params.dm_pulse_width  = 40;//e-6; // TODO: Check why this was here
    params.dm_nbits        = 32;//8;
    params.use_scrunching  = false;
    params.scrunch_tol     = 1.15;
    params.rfi_tol         = 5.0;//1e-6;//1e-9; TODO: Should this be a probability instead?
    params.rfi_min_beams   = 8;
    params.boxcar_max      = 4096;//2048;//512;
    params.detect_thresh   = 6.0;
    params.cand_sep_time   = 3;
    // Note: These have very little effect on the candidates, but could be important
    //         to capture (*rare*) coincident events.
    params.cand_sep_filter = 3;  // Note: filter numbers, not actual width
    params.cand_sep_dm     = 200; // Note: trials, not actual DM
    params.cand_rfi_dm_cut = 1.5;
    //params.cand_min_members = 3;

    // TODO: This still needs tuning!
    params.max_giant_rate  = 0;      // Max allowed giants per minute, 0 == no limit

    params.min_tscrunch_width = 4096; // Filter width at which to begin tscrunching

    params.num_channel_zaps = 0;
    params.channel_zaps = NULL;

    params.coincidencer_host = NULL;
    params.coincidencer_port = -1;

    // TESTING
    //params.first_beam = 0;
    params.beam_count = 1;
    params.gpu_id = 0;
    params.utc_start = 0;
    params.spectra_per_second = 0;
    params.output_dir = ".";
}

#endif
