#ifndef _H_PAFRB_CONFIG
#define _H_PAFRB_CONFIG

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <heimdall/params.hpp>

struct InConfig {
    bool test;
    bool verbose;

    double band;                // sampling rate for each band in MHz
    double dend;
    double dstart;
    double foff;                // channel width in MHz
    double ftop;                // frequency of the top channel in MHz
    double toread;
    double tsamp;               // sampling time


    std::string outdir;

    std::vector<int> gpuids;
    std::vector<std::string> ips;
    std::vector<int> killmask;
    std::vector<std::vector<int>> ports;

    unsigned int accumulate;    // number of 108us complete chunks to process on the GPU at once
    unsigned int batch;
    unsigned int beamno;        // number of beams per card
    unsigned int chunks;        // time chunks to process - testing only
    unsigned int fftsize;
    unsigned int filchans;      // number fo filterbank channels
    unsigned int freqavg;          // number of frequency channels to average
    unsigned int gulp;
    unsigned int headlen;
    unsigned int inbits;
    unsigned int nchans;        // number of 1MHz channels
    unsigned int ngpus;         // number of GPUs to use
    unsigned int npol;
    unsigned int vdiflen;   // length in bytes of the single VDIF packed excluding header
    unsigned int port;
    unsigned int stokes;        // number of Stokes parameters to output
    unsigned int streamno;      // number of CUDA streams for filterbanking
    unsigned int timesavg;         // number of time samples to average

};

inline void default_config(InConfig &config) {
    config.test = false;
    config.verbose = false;

    config.accumulate = 1024;
    config.band = 1.185;
    config.dstart = 0.0;
    config.dend = 4000.0;
    config.ftop = 1400.0;

    config.beamno = 1;
    config.chunks = 32;
    config.fftsize = 128;
    config.freqavg = 1;
    config.foff = (double)1.0/(double)27.0 * (double)config.freqavg;
    config.gulp = 131072;     // 2^17, equivalent to ~14s for 108us sampling time
    config.headlen = 32;
    config.inbits = 2;
    config.ngpus = 1;
    config.npol = 2;
    config.outdir = "/data/local/scratch/mat_test/";
    config.stokes = 4;
    config.timesavg = 1;
    config.vdiflen = 8000;

    config.batch = config.nchans;
    config.tsamp = (double)1.0 / (config.band * 1e+06) * 32 * (double)config.timesavg;
    for (int ii = 0; ii < config.filchans; ii++)
         (config.killmask).push_back((int)1);
}

inline void print_config(const InConfig &config) {

    // polymorphic lambda needs g++ version 4.9 or higher
    // std::vector<auto> might not work on all systems
    // no support for C++14 with nvcc yet

    auto plambda_s = [](std::ostream &os, std::string sintro, std::vector<std::string> values) -> std::ostream& {
        os << sintro << ": ";
        for (auto iptr = values.begin(); iptr != values.end(); iptr++)
            os << *iptr << " ";
        os << std::endl;
        return os;
    };

    auto plambda_i = [](std::ostream &os, std::string sintro, std::vector<int> values) -> std::ostream& {
        os << sintro << ": ";
        for (auto iptr = values.begin(); iptr != values.end(); iptr++)
            os << *iptr << " ";
        os << std::endl;
        return os;
    };

    std::cout << "Configuration overview: " << std::endl;
    std::cout << "\tNumber of GPUs to use: " << config.ngpus << std::endl;
    plambda_i(std::cout, "\t\tIDs", config.gpuids);
    std::cout << "\tNumber of CUDA streams used for filterbanking: " << config.streamno << std::endl;
    plambda_s(std::cout, "\tIPs to use", config.ips);
    std::cout << "\tPorts to use";
    for (int ii = 0; ii < config.ports.size(); ii++)
        plambda_i(std::cout, "\t\t" + config.ips[ii], config.ports[ii]);
    std::cout << "\tOutput directory: " << config.outdir << std::endl;
    std::cout << "\tNumber of generater filterbank channels: " << config.filchans << std::endl;
    std::cout << "\tNumber of channels to average: " << config.freqavg << std::endl;
    std::cout << "\tNumber of time samples to average:" << config.timesavg << std::endl;
    std::cout << "!!!CURRENTLY NOT IN USE!!!: " << std::endl;
    std::cout << "\tDedisperse gulp size: " << config.gulp << std::endl;
    std::cout << "\tStart DM: " << config.dstart << std::endl;
    std::cout << "\tEnd DM: " << config.dend << std::endl;

}

inline void read_config(std::string filename, InConfig &config) {

    std::fstream inconfig(filename.c_str(), std::ios_base::in);
    std::string line;
    std::string paraname;
    std::string paravalue;

    if(inconfig) {
        while(std::getline(inconfig, line)) {
            std::istringstream ossline(line);
            ossline >> paraname >> paravalue;

            if (paraname == "DM_END") {
                config.dend = stod(paravalue);
            } else if (paraname == "DM_START") {
                config.dstart = stod(paravalue);
            } else if (paraname == "FFT_SIZE") {
                config.fftsize = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "FREQ_AVERAGE") {
                config.freqavg = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "DEDISP_GULP") {
                config.gulp = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "GPU_IDS") {
                std::istringstream svalue(paravalue);
                std::string strgpu;
                while(std::getline(svalue, strgpu, ','))
                    config.gpuids.push_back(std::stoi(strgpu));
            } else if (paraname == "IP") {
                std::istringstream svalue(paravalue);
                std::string strip;
                while(std::getline(svalue, strip, ','))
                    config.ips.push_back(strip);
            } else if (paraname == "NO_1MHZ_CHANS") {
                config.nchans = (unsigned int)(std::stoi(paravalue));
                config.batch = config.nchans;
            } else if (paraname == "NO_BEAMS") {
                config.beamno = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "NO_GPUS") {
                config.ngpus = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "NO_POLS") {
                config.npol = std::stoi(paravalue);
            } else if (paraname == "NO_STOKES") {
                config.stokes = std::stoi(paravalue);
            } else if (paraname == "NO_STREAMS") {
                config.streamno = (unsigned int)(std::stoi(paravalue));
            } else if (paraname == "PORTS") {
                std::istringstream ssvalue(paravalue);
                std::string ipports;
                while(std::getline(ssvalue, ipports, ';')) {
                    std::vector<int> vtmp;
                    std::istringstream ssports(ipports);
                    std::string singleport;
                    while(std::getline(ssports, singleport, ','))
                        vtmp.push_back(stoi(singleport));

                    config.ports.push_back(vtmp);
                }
            } else if (paraname == "READ") {
                config.toread = std::stod(paravalue);
            } else if (paraname == "TIME_AVERAGE") {
                config.timesavg = (unsigned int)(std::stoi(paravalue));
            } else {
                std::cout << "Error: unrecognised parameter: " << paraname << std::endl;
            }
        }
    } else {
        std::cout << "Error opening the configuration file!!\n Will use default configuration instead." << std::endl;
    }

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
    params.dm_min          = config.dstart;
    params.dm_max          = config.dend;
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
