#ifndef _H_PAFRB_CONFIG
#define _H_PAFRB_CONFIG

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <heimdall/params.hpp>

using std::cout;
using std::endl;
using std::string;
using std::stod;
using std::stoi;

struct config_s {
    bool test;
    bool verbose;

    double band;                // sampling rate for each band in MHz
    double dend;
    double dstart;
    double foff;                // channel width in MHz
    double ftop;                // frequency of the top channel in MHz
    double tsamp;               // sampling time

    std::string outdir;

    std::vector<int> gpuids;
    std::vector<std::string> ips;
    std::vector<int> killmask;

    unsigned int accumulate;    // number of 108us complete chunks to process on the GPU at once
    unsigned int batch;
    unsigned int beamno;        // number of beams per card
    unsigned int chunks;        // time chunks to process - testing only
    unsigned int fftsize;
    unsigned int filchans;      // number fo filterbank channels
    unsigned int freqavg;          // number of frequency channels to average
    unsigned int gulp;
    unsigned int nchans;        // number of 1MHz channels
    unsigned int ngpus;         // number of GPUs to use
    unsigned int npol;
    unsigned int port;
    unsigned int stokes;        // number of Stokes parameters to output
    unsigned int streamno;      // number of CUDA streams for filterbanking
    unsigned int timesavg;         // number of time samples to average

};

inline void default_config(config_s &config) {
    config.test = false;
    config.verbose = false;

    config.accumulate = 8;
    config.band = 1.185;
    config.dstart = 0.0;
    config.dend = 4000.0;
    config.ftop = 1400.0;

    config.beamno = 1;
    config.chunks = 32;
    config.fftsize = 32;
    config.freqavg = 16;
    config.foff = (double)1.0/(double)27.0 * (double)config.freqavg;
    //config.gulp = 16384;        // 2^24, equivalent to ~1.75s for 108us sampling time (for testing purposes)
    config.gulp = 131072;     // 2^17, equivalent to ~14s for 108us sampling time
    // TEST
    //config.nchans = 42;
    config.nchans = 336;
    config.ngpus = 1;
    config.npol = 2;
    config.outdir = "/data/local/scratch/mat_test/";
    config.stokes = 4;
    config.streamno = 4;
    config.timesavg = 4;

    config.batch = config.nchans;
    config.filchans = config.nchans * 27 / config.freqavg;
    config.tsamp = (double)1.0 / (config.band * 1e+06) * 32 * (double)config.timesavg;
    for (int ii = 0; ii < config.filchans; ii++)
         (config.killmask).push_back((int)1);
}

inline void read_config(string filename, config_s &config) {

    std::fstream inconfig(filename.c_str(), std::ios_base::in);
    string line;
    string paraname;
    string paravalue;

    if(inconfig) {
        while(std::getline(inconfig, line)) {
            std::istringstream ossline(line);
            ossline >> paraname >> paravalue;
            std::stringstream svalue;

            if (paraname == "DM_END") {
                config.dend = stod(paravalue);
            } else if (paraname == "DM_START") {
                config.dstart = stod(paravalue);
            } else if (paraname == "FFT_SIZE") {
                config.fftsize = (unsigned int)(stoi(paravalue));
            } else if (paraname == "FREQ_AVERAGE") {
                config.freqavg = (unsigned int)(stoi(paravalue));
            } else if (paraname == "DEDISP_GULP") {
                config.gulp = (unsigned int)(stoi(paravalue));
            } else if (paraname == "GPU_IDS") {
                std::stringstream svalue(paravalue);
                string sep;
                while(std::getline(svalue, sep, ','))
                    config.gpuids.push_back(stoi(sep));
            } else if (paraname == "IP") {
                std::stringstream svalue(paravalue);
                string sep;
                while(std::getline(svalue, sep, ','))
                    config.ips.push_back(sep);
            } else if (paraname == "NO_1MHZ_CHANS") {
                config.nchans = (unsigned int)(stoi(paravalue));
                config.batch = config.nchans;
            } else if (paraname == "NO_BEAMS") {
                config.beamno = (unsigned int)(stoi(paravalue));
            } else if (paraname == "NO_GPUS") {
                config.ngpus = (unsigned int)(stoi(paravalue));
            } else if (paraname == "NO_POLS") {
                config.npol = stoi(paravalue);
            } else if (paraname == "NO_STOKES") {
                config.stokes = stoi(paravalue);
            } else if (paraname == "NO_STREAMS") {
                config.streamno = (unsigned int)(stoi(paravalue));
            } else if (paraname == "TIME_AVERAGE") {
                config.timesavg = (unsigned int)(stoi(paravalue));
            } else {
                cout << "Error: unrecognised parameter: " << paraname << endl;
            }
        }
    } else {
        cout << "Error opening the configuration file!!\n Will use default configuration instead." << endl;
    }

    inconfig.close();
}

inline void set_search_params(hd_params &params, config_s config)
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
