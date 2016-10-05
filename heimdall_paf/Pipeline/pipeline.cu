/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include <vector>
#include <memory>
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <sstream>
#include <iomanip>
#include <string>
#include <fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using thrust::host_vector;
using thrust::device_vector;
#include <thrust/version.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>

#include "hd/pipeline.h"
#include "hd/maths.h"
#include "hd/clean_filterbank_rfi.h"

#include "hd/remove_baseline.h"
#include "hd/matched_filter.h"
#include "hd/get_rms.h"
#include "hd/find_giants.h"
#include "hd/label_candidate_clusters.h"
#include "hd/merge_candidates.h"

#include "hd/DataSource.h"
#include "hd/ClientSocket.h"
#include "hd/SocketException.h"
#include "hd/stopwatch.h"         // For benchmarking
//#include "write_time_series.h" // For debugging

#include <dedisp.h>

#define HD_BENCHMARK

#ifdef HD_BENCHMARK
  void start_timer(Stopwatch& timer) { timer.start(); }
  void stop_timer(Stopwatch& timer) { cudaThreadSynchronize(); timer.stop(); }
#else
  void start_timer(Stopwatch& timer) { }
  void stop_timer(Stopwatch& timer) { }
#endif // HD_BENCHMARK

#include <utility> // For std::pair
template<typename T, typename U>
std::pair<T&,U&> tie(T& a, U& b) { return std::pair<T&,U&>(a,b); }

struct hd_pipeline_t {
  hd_params   params;
  dedisp_plan dedispersion_plan;
  //MPI_Comm    communicator;

  // Memory buffers used during pipeline execution
  std::vector<hd_byte>    h_clean_filterbank;
  host_vector<hd_byte>    h_dm_series;
  device_vector<hd_float> d_time_series;
  device_vector<hd_float> d_filtered_series;
};

unsigned int get_filter_index(unsigned int filter_width) {
  // This function finds log2 of the 32-bit power-of-two number v
  unsigned int v = filter_width;
  static const unsigned int b[] = {0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0,
                                   0xFF00FF01, 0xFFFF0000};
  register unsigned int r = (v & b[0]) != 0;
  for( int i=4; i>0; --i) {
    r |= ((v & b[i]) != 0) << i;
  }
  return r;
}

// now this looks like very much not needed function
// will need to change the hd_pipeline structure
hd_error hd_create_pipeline(hd_pipeline* pipeline_, hd_params params) {
  *pipeline_ = 0;

  // Note: We use a smart pointer here to automatically clean up after errors
  typedef std::auto_ptr<hd_pipeline_t> smart_pipeline_ptr;
  smart_pipeline_ptr pipeline = smart_pipeline_ptr(new hd_pipeline_t());
  if( !pipeline.get() ) {
    return throw_error(HD_MEM_ALLOC_FAILED);
  }

  pipeline->params = params;

  *pipeline_ = pipeline.release();

}
// unsigned char *d_dedisp is the pointer to the dedispersed data residing on the device memory
hd_error hd_execute(hd_pipeline pl, unsigned char *d_dedisp, hd_size nsamps,
                        hd_size nbits, unsigned int gulps_processed)

hd_error hd_execute(hd_pipeline pl,
                    const hd_byte* h_filterbank, hd_size nsamps, hd_size nbits,
                    hd_size first_idx, hd_size* nsamps_processed) {

      hd_error error = HD_NO_ERROR;

      hd_size nbytes = nsamps * pl->params.nchans * nbits / 8;

      pl->h_clean_filterbank.resize(nbytes);
      std::vector<int>          h_killmask(pl->params.nchans, 1);

      hd_size      dm_count = dedisp_get_dm_count(pl->dedispersion_plan);
      const float* dm_list  = dedisp_get_dm_list(pl->dedispersion_plan);

      hd_size nsamps_computed  = nsamps;
      hd_size series_stride    = nsamps_computed;

      const dedisp_size *scrunch_factors = dedisp_get_dt_factors(pl->dedispersion_plan);

      // Report the number of samples that will be properly processed
      *nsamps_processed = nsamps_computed - pl->params.boxcar_max;

      pl->h_dm_series.resize(series_stride * pl->params.dm_nbits/8 * dm_count);
      pl->d_time_series.resize(series_stride);
      pl->d_filtered_series.resize(series_stride, 0);

      RemoveBaselinePlan          baseline_remover;
      GetRMSPlan                  rms_getter;
      MatchedFilterPlan<hd_float> matched_filter_plan;
      GiantFinder                 giant_finder;

      thrust::device_vector<hd_float> d_giant_peaks;
      thrust::device_vector<hd_size>  d_giant_inds;
      thrust::device_vector<hd_size>  d_giant_begins;
      thrust::device_vector<hd_size>  d_giant_ends;
      thrust::device_vector<hd_size>  d_giant_filter_inds;
      thrust::device_vector<hd_size>  d_giant_dm_inds;
      thrust::device_vector<hd_size>  d_giant_members;

      typedef thrust::device_ptr<hd_float> dev_float_ptr;
      typedef thrust::device_ptr<hd_size>  dev_size_ptr;

      // TESTING
      hd_size write_dm = 0;

      bool too_many_giants = false;

      // For each DM
      for( hd_size dm_idx=0; dm_idx<dm_count; ++dm_idx ) {
          hd_size  cur_dm_scrunch = scrunch_factors[dm_idx];
          hd_size  cur_nsamps  = nsamps_computed / cur_dm_scrunch;
          hd_float cur_dt      = pl->params.dt * cur_dm_scrunch;

            // Bail if the candidate rate is too high
            if( too_many_giants ) {
                break;
            }


        hd_float* time_series = thrust::raw_pointer_cast(&pl->d_time_series[0]);

        // Copy the time series to the device and convert to floats
        hd_size offset = dm_idx * series_stride * pl->params.dm_nbits/8;
        switch( pl->params.dm_nbits ) {
            case 8:
                thrust::device_vector<float> d_time_series((unsigned char*)d_dedisp, (unsigned char*)d_dedisp + offset);
            break;
            case 16:
                thrust::device_vector<float> d_time_series((unsigned short*)d_dedisp, (unsigned short*)d_dedisp + offset);
            break;
            case 32:
                // Note: 32-bit implies float, not unsigned int
                thrust::device_vector<float> d_time_series((float*)d_dedisp, (float*)d_dedisp + offset);
            break;
            default:
                return HD_INVALID_NBITS;
        }

        // Remove the baseline
        // -------------------
        // Note: Divided by 2 to form a smoothing radius
        hd_size nsamps_smooth = hd_size(pl->params.baseline_length /
                                        (2 * cur_dt));
        // Crop the smoothing length in case not enough samples

        error = baseline_remover.exec(time_series, cur_nsamps, nsamps_smooth);
        if( error != HD_NO_ERROR ) {
            return throw_error(error);
        }

    // Normalise
    // ---------
    hd_float rms = rms_getter.exec(time_series, cur_nsamps);
    thrust::transform(pl->d_time_series.begin(), pl->d_time_series.end(),
                      thrust::make_constant_iterator(hd_float(1.0)/rms),
                      pl->d_time_series.begin(),
                      thrust::multiplies<hd_float>());

    // Prepare the boxcar filters
    // --------------------------
    // We can't process the first and last max-filter-width/2 samples
    hd_size rel_boxcar_max = pl->params.boxcar_max/cur_dm_scrunch;

    hd_size max_nsamps_filtered = cur_nsamps + 1 - rel_boxcar_max;
    // This is the relative offset into the time series of the filtered data
    hd_size cur_filtered_offset = rel_boxcar_max / 2;

    // Create and prepare matched filtering operations
    // Note: Filter width is relative to the current time resolution
    matched_filter_plan.prep(time_series, cur_nsamps, rel_boxcar_max);

    hd_float* filtered_series = thrust::raw_pointer_cast(&pl->d_filtered_series[0]);

    // Note: Filtering is done using a combination of tscrunching and
    //         'proper' boxcar convolution. The parameter min_tscrunch_width
    //         indicates how much of each to do. Raising min_tscrunch_width
    //         increases sensitivity but decreases performance and vice
    //         versa.

    // For each boxcar filter
    // Note: We cannot detect pulse widths < current time resolution
    for( hd_size filter_width=cur_dm_scrunch;
         filter_width<=pl->params.boxcar_max;
         filter_width*=2 ) {
      hd_size rel_filter_width = filter_width / cur_dm_scrunch;
      hd_size filter_idx = get_filter_index(filter_width);

      // Note: Filter width is relative to the current time resolution
      hd_size rel_min_tscrunch_width = std::max(pl->params.min_tscrunch_width
                                                / cur_dm_scrunch,
                                                hd_size(1));
      hd_size rel_tscrunch_width = std::max(2 * rel_filter_width
                                            / rel_min_tscrunch_width,
                                            hd_size(1));
      // Filter width relative to cur_dm_scrunch AND tscrunch
      hd_size rel_rel_filter_width = rel_filter_width / rel_tscrunch_width;

      error = matched_filter_plan.exec(filtered_series,
                                       rel_filter_width,
                                       rel_tscrunch_width);

      if( error != HD_NO_ERROR ) {
        return throw_error(error);
      }
      // Divide and round up
      hd_size cur_nsamps_filtered = ((max_nsamps_filtered-1)
                                     / rel_tscrunch_width + 1);
      hd_size cur_scrunch = cur_dm_scrunch * rel_tscrunch_width;

      // Normalise the filtered time series (RMS ~ sqrt(time))
      // TODO: Avoid/hide the ugly thrust code?
      //         Consider making it a method of MatchedFilterPlan
      /*
      thrust::constant_iterator<hd_float>
        norm_val_iter(1.0 / sqrt((hd_float)rel_filter_width));
      thrust::transform(thrust::device_ptr<hd_float>(filtered_series),
                        thrust::device_ptr<hd_float>(filtered_series)
                        + cur_nsamps_filtered,
                        norm_val_iter,
                        thrust::device_ptr<hd_float>(filtered_series),
                        thrust::multiplies<hd_float>());
      */
      // TESTING Proper normalisation
      hd_float rms = rms_getter.exec(filtered_series, cur_nsamps_filtered);
      thrust::transform(thrust::device_ptr<hd_float>(filtered_series),
                        thrust::device_ptr<hd_float>(filtered_series)
                        + cur_nsamps_filtered,
                        thrust::make_constant_iterator(hd_float(1.0)/rms),
                        thrust::device_ptr<hd_float>(filtered_series),
                        thrust::multiplies<hd_float>());

      hd_size prev_giant_count = d_giant_peaks.size();

      error = giant_finder.exec(filtered_series, cur_nsamps_filtered,
                                pl->params.detect_thresh,
                                //pl->params.cand_sep_time,
                                // Note: This was MB's recommendation
                                pl->params.cand_sep_time * rel_rel_filter_width,
                                d_giant_peaks,
                                d_giant_inds,
                                d_giant_begins,
                                d_giant_ends);

      if( error != HD_NO_ERROR ) {
        return throw_error(error);
      }

      hd_size rel_cur_filtered_offset = (cur_filtered_offset /
                                         rel_tscrunch_width);

      using namespace thrust::placeholders;
      thrust::transform(d_giant_inds.begin()+prev_giant_count,
                        d_giant_inds.end(),
                        d_giant_inds.begin()+prev_giant_count,
                        /*first_idx +*/ (_1+rel_cur_filtered_offset)*cur_scrunch);
      thrust::transform(d_giant_begins.begin()+prev_giant_count,
                        d_giant_begins.end(),
                        d_giant_begins.begin()+prev_giant_count,
                        /*first_idx +*/ (_1+rel_cur_filtered_offset)*cur_scrunch);
      thrust::transform(d_giant_ends.begin()+prev_giant_count,
                        d_giant_ends.end(),
                        d_giant_ends.begin()+prev_giant_count,
                        /*first_idx +*/ (_1+rel_cur_filtered_offset)*cur_scrunch);

      d_giant_filter_inds.resize(d_giant_peaks.size(), filter_idx);
      d_giant_dm_inds.resize(d_giant_peaks.size(), dm_idx);
      // Note: This could be used to track total member samples if desired
      d_giant_members.resize(d_giant_peaks.size(), 1);

      // Bail if the candidate rate is too high
      hd_size total_giant_count = d_giant_peaks.size();
      hd_float data_length_mins = nsamps * pl->params.dt / 60.0;
      if ( pl->params.max_giant_rate && ( total_giant_count / data_length_mins > pl->params.max_giant_rate ) ) {
        too_many_giants = true;
        float searched = ((float) dm_idx * 100) / (float) dm_count;
        cout << "WARNING: exceeded max giants/min, DM [" << dm_list[dm_idx] << "] space searched " << searched << "%" << endl;
        break;
      }

    } // End of filter width loop
  } // End of DM loop

  hd_size giant_count = d_giant_peaks.size();

  thrust::host_vector<hd_float> h_group_peaks;
  thrust::host_vector<hd_size>  h_group_inds;
  thrust::host_vector<hd_size>  h_group_begins;
  thrust::host_vector<hd_size>  h_group_ends;
  thrust::host_vector<hd_size>  h_group_filter_inds;
  thrust::host_vector<hd_size>  h_group_dm_inds;
  thrust::host_vector<hd_size>  h_group_members;
  thrust::host_vector<hd_float> h_group_dms;

  //if (!too_many_giants)
  //{
    thrust::device_vector<hd_size> d_giant_labels(giant_count);
    hd_size* d_giant_labels_ptr = thrust::raw_pointer_cast(&d_giant_labels[0]);

    RawCandidates d_giants;
    d_giants.peaks = thrust::raw_pointer_cast(&d_giant_peaks[0]);
    d_giants.inds = thrust::raw_pointer_cast(&d_giant_inds[0]);
    d_giants.begins = thrust::raw_pointer_cast(&d_giant_begins[0]);
    d_giants.ends = thrust::raw_pointer_cast(&d_giant_ends[0]);
    d_giants.filter_inds = thrust::raw_pointer_cast(&d_giant_filter_inds[0]);
    d_giants.dm_inds = thrust::raw_pointer_cast(&d_giant_dm_inds[0]);
    d_giants.members = thrust::raw_pointer_cast(&d_giant_members[0]);

    hd_size filter_count = get_filter_index(pl->params.boxcar_max) + 1;

    hd_size label_count;
    error = label_candidate_clusters(giant_count,
                                     *(ConstRawCandidates*)&d_giants,
                                     pl->params.cand_sep_time,
                                     pl->params.cand_sep_filter,
                                     pl->params.cand_sep_dm,
                                     d_giant_labels_ptr,
                                     &label_count);
    if( error != HD_NO_ERROR ) {
      return throw_error(error);
    }

    hd_size group_count = label_count;

    thrust::device_vector<hd_float> d_group_peaks(group_count);
    thrust::device_vector<hd_size>  d_group_inds(group_count);
    thrust::device_vector<hd_size>  d_group_begins(group_count);
    thrust::device_vector<hd_size>  d_group_ends(group_count);
    thrust::device_vector<hd_size>  d_group_filter_inds(group_count);
    thrust::device_vector<hd_size>  d_group_dm_inds(group_count);
    thrust::device_vector<hd_size>  d_group_members(group_count);

    thrust::device_vector<hd_float> d_group_dms(group_count);

    RawCandidates d_groups;
    d_groups.peaks = thrust::raw_pointer_cast(&d_group_peaks[0]);
    d_groups.inds = thrust::raw_pointer_cast(&d_group_inds[0]);
    d_groups.begins = thrust::raw_pointer_cast(&d_group_begins[0]);
    d_groups.ends = thrust::raw_pointer_cast(&d_group_ends[0]);
    d_groups.filter_inds = thrust::raw_pointer_cast(&d_group_filter_inds[0]);
    d_groups.dm_inds = thrust::raw_pointer_cast(&d_group_dm_inds[0]);
    d_groups.members = thrust::raw_pointer_cast(&d_group_members[0]);

    merge_candidates(giant_count,
                     d_giant_labels_ptr,
                     *(ConstRawCandidates*)&d_giants,
                     d_groups);

    // Look up the actual DM of each group
    thrust::device_vector<hd_float> d_dm_list(dm_list, dm_list+dm_count);
    thrust::gather(d_group_dm_inds.begin(), d_group_dm_inds.end(),
                   d_dm_list.begin(),
                   d_group_dms.begin());

    // Device to host transfer of candidates
    h_group_peaks = d_group_peaks;
    h_group_inds = d_group_inds;
    h_group_begins = d_group_begins;
    h_group_ends = d_group_ends;
    h_group_filter_inds = d_group_filter_inds;
    h_group_dm_inds = d_group_dm_inds;
    h_group_members = d_group_members;
    h_group_dms = d_group_dms;
    //h_group_flags = d_group_flags;
  //}

  char buffer[64];
  time_t now = pl->params.utc_start + (time_t) (first_idx / pl->params.spectra_per_second);
  strftime (buffer, 64, HD_TIMESTR, (struct tm*) gmtime(&now));

  std::stringstream ss;
  ss << std::setw(2) << std::setfill('0') << (pl->params.beam)%13+1;

  std::ostringstream oss;

  if ( pl->params.coincidencer_host != NULL && pl->params.coincidencer_port != -1 )
  {
    try
    {
      ClientSocket client_socket ( pl->params.coincidencer_host, pl->params.coincidencer_port );

      strftime (buffer, 64, HD_TIMESTR, (struct tm*) gmtime(&(pl->params.utc_start)));

      oss <<  buffer << " ";

      time_t now = pl->params.utc_start + (time_t) (first_idx / pl->params.spectra_per_second);
      strftime (buffer, 64, HD_TIMESTR, (struct tm*) gmtime(&now));
      oss << buffer << " ";

      oss << first_idx << " ";
      oss << ss.str() << " ";
      oss << h_group_peaks.size() << endl;
      client_socket << oss.str();
      oss.flush();
      oss.str("");

      for (hd_size i=0; i<h_group_peaks.size(); ++i )
      {
        hd_size samp_idx = first_idx + h_group_inds[i];
        oss << h_group_peaks[i] << "\t"
                      << samp_idx << "\t"
                      << samp_idx * pl->params.dt << "\t"
                      << h_group_filter_inds[i] << "\t"
                      << h_group_dm_inds[i] << "\t"
                      << h_group_dms[i] << "\t"
                      << h_group_members[i] << "\t"
                      << first_idx + h_group_begins[i] << "\t"
                      << first_idx + h_group_ends[i] << endl;

        client_socket << oss.str();
        oss.flush();
        oss.str("");
      }
      // client_socket should close when it goes out of scope...
    }
    catch (SocketException& e )
    {
      std::cerr << "SocketException was caught:" << e.description() << "\n";
    }

  }
  //else
  //{

    std::string filename = std::string(pl->params.output_dir) + "/" + std::string(buffer) + "_" + ss.str() + ".cand";
    std::ofstream cand_file(filename.c_str(), std::ios::out);

    if (cand_file.good())
    {
      for( hd_size i=0; i<h_group_peaks.size(); ++i ) {
        hd_size samp_idx = first_idx + h_group_inds[i];
        cand_file << h_group_peaks[i] << "\t"
                  << samp_idx << "\t"
                  << samp_idx * pl->params.dt << "\t"
                  << h_group_filter_inds[i] << "\t"
                  << h_group_dm_inds[i] << "\t"
                  << h_group_dms[i] << "\t"
                  //<< h_group_flags[i] << "\t"
                  << h_group_members[i] << "\t"
                  // HACK %13
                  //<< (beam+pl->params.beam)%13+1 << "\t"
                  << first_idx + h_group_begins[i] << "\t"
                  << first_idx + h_group_ends[i] << "\t"
                  << "\n";
      }
    }
    else
      cout << "Skipping dump due to bad file open on " << filename << endl;
    cand_file.close();
  //}

  if( too_many_giants ) {
    return HD_TOO_MANY_EVENTS;
  }
  else {
    return HD_NO_ERROR;
  }
}

void hd_destroy_pipeline(hd_pipeline pipeline) {

  dedisp_destroy_plan(pipeline->dedispersion_plan);

  // Note: This assumes memory owned by pipeline cleans itself up
  if( pipeline ) {
    delete pipeline;
  }
}
