/*
 *  Copyright 2012 Ben Barsdell
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
  This file contains the boring boiler-plate code to manage the library.

  TODO: Test on 32-bit integer input
        Consider accepting 32-bit floats instead of 32-bit ints
*/

//#define DEDISP_DEBUG
//#define DEDISP_BENCHMARK

#include "errors.hpp"
#include "dedisp/dedisp.hpp"
#include <pthread.h>

#include <vector>
#include <algorithm> // For std::fill

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
// For copying and scrunching the DM list
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>

#ifdef DEDISP_BENCHMARK
#include <fstream>
#endif

#if defined(DEDISP_DEBUG) && DEDISP_DEBUG
#include <stdio.h> //For printf
#endif

#include <iostream>
using std::cout;
using std::endl;

#include <dedisp/gpu_memory.hpp>
#include <dedisp/transpose.hpp>

#define DEDISP_DEFAULT_GULP_SIZE 65536 //131072

// Note: The implementation of the sub-band algorithm is a prototype only
//         Enable at your own risk! It may not be in a working state at all.
//#define USE_SUBBAND_ALGORITHM
#define DEDISP_DEFAULT_SUBBAND_SIZE 32

// TODO: Make sure this doesn't limit GPU constant memory
//         available to users.
#define DEDISP_MAX_NCHANS 8192
// Internal word type used for transpose and dedispersion kernel
typedef unsigned int dedisp_word;
// Note: This must be included after the above #define and typedef
#include <dedisp/kernels.cuh>

// Define plan structure
struct dedisp_plan_struct {
  // Multi-GPU parameters
  dedisp_size  device_count;
  // Size parameters
  dedisp_size  dm_count;
  dedisp_size  nchans;
  dedisp_size  max_delay;
  dedisp_size  gulp_size;
  // Physical parameters
  dedisp_float dt;
  dedisp_float f0;
  dedisp_float df;
  // Host arrays
  std::vector<dedisp_float> dm_list;      // size = dm_count
  std::vector<dedisp_float> delay_table;  // size = nchans
  std::vector<dedisp_bool>  killmask;     // size = nchans
  std::vector<dedisp_size>  scrunch_list; // size = dm_count
  // Device arrays //NEW: one for each GPU
  std::vector< thrust::device_vector<dedisp_float> > d_dm_list;
  std::vector< thrust::device_vector<dedisp_float> > d_delay_table;
  std::vector< thrust::device_vector<dedisp_bool> >  d_killmask;
  std::vector< thrust::device_vector<dedisp_size> >  d_scrunch_list;
  //StreamType stream;
  // Scrunching parameters
  dedisp_bool  scrunching_enabled;
  dedisp_float pulse_width;
  dedisp_float scrunch_tol;
  int gpuid;
    cudaStream_t stream;
};

//Thread argument container //NEW
struct dedisp_thread_args {
  //Standard dedisp_execute_guru args
  dedisp_plan plan;
  dedisp_size nsamps;
  dedisp_byte *in;
  dedisp_size in_nbits;
  dedisp_size in_stride;
  dedisp_byte *out;
  dedisp_size out_nbits;
  dedisp_size out_stride;
  unsigned flags;
  //Identifier for the device being used
  unsigned int device_idx;
};

// Private helper functions
// ------------------------
template<typename T>
T min(T a, T b) { return a<b ? a : b; }
unsigned long div_round_up(unsigned long a, unsigned long b) {
	return (a-1) / b + 1;
}

// Internal abstraction for errors
#if defined(DEDISP_DEBUG) && DEDISP_DEBUG
#define throw_error(error) do {                                         \
	printf("An error occurred within dedisp on line %d of %s: %s",      \
	       __LINE__, __FILE__, dedisp_get_error_string(error));         \
	return (error); } while(0)
#define throw_getter_error(error, retval) do {                          \
	printf("An error occurred within dedisp on line %d of %s: %s",      \
	       __LINE__, __FILE__, dedisp_get_error_string(error));         \
	return (retval); } while(0)
#else
#define throw_error(error) return error
#define throw_getter_error(error, retval) return retval
#endif // DEDISP_DEBUG
/*
dedisp_error throw_error(dedisp_error error) {
	// Note: Could, e.g., put an error callback in here
	return error;
}
*/

dedisp_error update_scrunch_list(dedisp_plan plan)
{
  if( cudaGetLastError() != cudaSuccess ) {
    throw_error(DEDISP_PRIOR_GPU_ERROR);
  }

  if( !plan->scrunching_enabled || 0 == plan->dm_count ) {
    plan->scrunch_list.resize(0);
    // Fill with 1's by default for safety
    plan->scrunch_list.resize(plan->dm_count, dedisp_size(1));
    return DEDISP_NO_ERROR;
  }

  plan->scrunch_list.resize(plan->dm_count);
  dedisp_error error = generate_scrunch_list(&plan->scrunch_list[0],
					     plan->dm_count,
					     plan->dt,
					     &plan->dm_list[0],
					     plan->nchans,
					     plan->f0,
					     plan->df,
					     plan->pulse_width,
					     plan->scrunch_tol);
  if( error != DEDISP_NO_ERROR ) {
    return error;
  }

  //NEW: Allocate on and copy to all devices
  for (int ii = 0; ii < plan->device_count; ii++)
    {
      error = dedisp_set_device(plan->gpuid);
      if (error != DEDISP_NO_ERROR)
	{
	  dedisp_destroy_plan(plan);
	  return error;
	}
      try {
	plan->d_scrunch_list[ii].resize(plan->dm_count);
      }
      catch(...) {
	throw_error(DEDISP_MEM_ALLOC_FAILED);
      }
      try {
	plan->d_scrunch_list[ii] = plan->scrunch_list;
      }
      catch(...) {
	throw_error(DEDISP_MEM_COPY_FAILED);
      }
    }
  return DEDISP_NO_ERROR;
}

dedisp_error set_requested_devices(dedisp_plan plan)
{
  int device_count;
  cudaGetDeviceCount(&device_count);
  if (plan->device_count > (dedisp_size) device_count)
    plan->device_count = (dedisp_size) device_count;
  //NEW: Not sure if any error checking is required here

  return DEDISP_NO_ERROR;
}

// ------------------------

// Public functions
// ----------------

dedisp_error dedisp_create_plan_multi(dedisp_plan* plan_,
				      dedisp_size  nchans,
				      dedisp_float dt,
				      dedisp_float f0,
				      dedisp_float df,
				      dedisp_size ngpus,
				      int gpuid)
{
  dedisp_error err;

  /// Initialise to NULL for safety
  *plan_ = 0;

  if( cudaGetLastError() != cudaSuccess ) {
    throw_error(DEDISP_PRIOR_GPU_ERROR);
  }

	//int device_idx; //NEW
	//cudaGetDevice(&device_idx);

	// Check for parameter errors
	if( nchans > DEDISP_MAX_NCHANS ) {
		throw_error(DEDISP_NCHANS_EXCEEDS_LIMIT);
	}

	// Force the df parameter to be negative such that
	//   freq[chan] = f0 + chan * df.
	df = -abs(df);

	dedisp_plan plan = new dedisp_plan_struct();
	if( !plan ) {
	  throw_error(DEDISP_MEM_ALLOC_FAILED);
	}

	plan->device_count  = ngpus;
	plan->dm_count      = 0;
	plan->nchans        = nchans;
	plan->gulp_size     = DEDISP_DEFAULT_GULP_SIZE;
	plan->max_delay     = 0;
	plan->dt            = dt;
	plan->f0            = f0;
	plan->df            = df;
    plan->gpuid         = gpuid;

	//NEW: resize containers for cuda device vectors
	plan->d_delay_table.resize(plan->device_count);
	plan->d_dm_list.resize(plan->device_count);
	plan->d_killmask.resize(plan->device_count);
	plan->d_scrunch_list.resize(plan->device_count);

	// Generate delay table and copy to device memory on each device
	// Note: The DM factor is left out and applied during dedispersion
	plan->delay_table.resize(plan->nchans);
	generate_delay_table(&plan->delay_table[0], plan->nchans, dt, f0, df);


	//NEW: moved this statement to keep a single loop below
	plan->killmask.resize(plan->nchans, (dedisp_bool)true);

	//NEW: resize the device vectors for each device
	for (int ii = 0; ii < plan->device_count; ii++)
	  {
	    err = dedisp_set_device(gpuid); 
	    if (err != DEDISP_NO_ERROR)
	      {
		dedisp_destroy_plan(plan);
		return err;
	      }
	    try {
	      plan->d_delay_table[ii].resize(plan->nchans);
	    }
	    catch(...) {
	      dedisp_destroy_plan(plan);
	      throw_error(DEDISP_MEM_ALLOC_FAILED);
	    }
	    try {
	      plan->d_delay_table[ii] = plan->delay_table;
	    }
	    catch(...) {
	      dedisp_destroy_plan(plan);
	      throw_error(DEDISP_MEM_COPY_FAILED);
	    }
	    try {
	      plan->d_killmask[ii].resize(plan->nchans);
	    }
	    catch(...) {
	      dedisp_destroy_plan(plan);
	      throw_error(DEDISP_MEM_ALLOC_FAILED);
	    }
	  }
	err = dedisp_set_killmask(plan, (dedisp_bool*)0);
	if( err != DEDISP_NO_ERROR ) {
	  dedisp_destroy_plan(plan);
	  return err;
	}
	*plan_ = plan;
	return DEDISP_NO_ERROR;
}

//NEW: for back compatibility
dedisp_error dedisp_create_plan(dedisp_plan* plan_,
				dedisp_size  nchans,
                                dedisp_float dt,
                                dedisp_float f0,
                                dedisp_float df,
                                int gpuid)
{
  return dedisp_create_plan_multi(plan_, nchans, dt, f0, df, 1, gpuid);
}

dedisp_error dedisp_set_gulp_size(dedisp_plan plan,
                                  dedisp_size gulp_size) {
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	plan->gulp_size = gulp_size;
	return DEDISP_NO_ERROR;
}
dedisp_size dedisp_get_gulp_size(dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->gulp_size;
}

dedisp_error dedisp_set_dm_list(dedisp_plan plan,
                                const dedisp_float* dm_list,
                                dedisp_size count)
{
        dedisp_error err;
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	if( !dm_list ) {
		throw_error(DEDISP_INVALID_POINTER);
	}
	if( cudaGetLastError() != cudaSuccess ) {
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	plan->dm_count = count;
	plan->dm_list.assign(dm_list, dm_list+count);

	// Copy to the device
	//NEW: copy to all devices
	for (int ii = 0; ii < plan->device_count; ii++)
	  {
	    err = dedisp_set_device(plan->gpuid);
	    if (err != DEDISP_NO_ERROR)
	      {
		dedisp_destroy_plan(plan);
		return err;
	      }
	    try {
	      plan->d_dm_list[ii].resize(plan->dm_count);
	    }
	    catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	    try {
	      plan->d_dm_list[ii] = plan->dm_list;
	    }
	    catch(...) { throw_error(DEDISP_MEM_COPY_FAILED); }
	  }
	// Calculate the maximum delay and store it in the plan
	plan->max_delay = dedisp_size(plan->dm_list[plan->dm_count-1] *
				      plan->delay_table[plan->nchans-1] + 0.5);

	err = update_scrunch_list(plan);
	if( err != DEDISP_NO_ERROR ) {
	  return err;
	}

	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_generate_dm_list(dedisp_plan plan,
                                     dedisp_float dm_start, dedisp_float dm_end,
                                     dedisp_float ti, dedisp_float tol)
{
        dedisp_error err;
        if( !plan ) {
	  throw_error(DEDISP_INVALID_PLAN);
	}
	if( cudaGetLastError() != cudaSuccess ) {
	  throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	// Generate the DM list (on the host)
	plan->dm_list.clear();
	generate_dm_list(plan->dm_list,
			 dm_start, dm_end,
			 plan->dt, ti, plan->f0, plan->df,
			 plan->nchans, tol);
	plan->dm_count = plan->dm_list.size();
	// Allocate device memory for the DM list
	//NEW: do this for all devices
    for (int ii = 0; ii < plan->device_count; ii++) {
        cudaCheckError(cudaSetDevice(plan->gpuid));
        /* if (err != DEDISP_NO_ERROR)
              {
                dedisp_destroy_plan(plan);
                return err;
            } */
	    try {
            plan->d_dm_list[ii].resize(plan->dm_count);
	    }
	    catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	    try {
            plan->d_dm_list[ii] = plan->dm_list;
	    }
	    catch(...) { throw_error(DEDISP_MEM_COPY_FAILED); }
    }

	// Calculate the maximum delay and store it in the plan
	plan->max_delay = dedisp_size(plan->dm_list[plan->dm_count-1] *
				      plan->delay_table[plan->nchans-1] + 0.5);

	err = update_scrunch_list(plan);
	if( err != DEDISP_NO_ERROR ) {
	  return err;
	}

	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_set_device(int device_idx) {
	if( cudaGetLastError() != cudaSuccess ) {
	  throw_error(DEDISP_PRIOR_GPU_ERROR);
	}

	cudaError_t error = cudaSetDevice(device_idx);
	// Note: cudaErrorInvalidValue isn't a documented return value, but
	//         it still gets returned :/
	if( cudaErrorInvalidDevice == error ||
		cudaErrorInvalidValue == error )
		throw_error(DEDISP_INVALID_DEVICE_INDEX);
	else if( cudaErrorSetOnActiveProcess == error )
		throw_error(DEDISP_DEVICE_ALREADY_SET);
	else if( cudaSuccess != error )
		throw_error(DEDISP_UNKNOWN_ERROR);
	else
		return DEDISP_NO_ERROR;
}

dedisp_error dedisp_set_killmask(dedisp_plan plan, const dedisp_bool* killmask)
{
  //NEW: Added loops to copy to all devices
  dedisp_error err;
  int ii;
  if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
  if( cudaGetLastError() != cudaSuccess ) {
    throw_error(DEDISP_PRIOR_GPU_ERROR);
  }
  if( 0 != killmask ) {
    // Copy killmask to plan (both host and device)

    plan->killmask.assign(killmask, killmask + plan->nchans);

    for (ii = 0; ii < plan->device_count; ii++)
      {
	err = dedisp_set_device(plan->gpuid);
	if (err != DEDISP_NO_ERROR)
	  {
	    dedisp_destroy_plan(plan);
	    return err;
	  }

	try {
	  plan->d_killmask[ii] = plan->killmask;
	}
	catch(...) { throw_error(DEDISP_MEM_COPY_FAILED); }
      }
  }
  else {
    // Set the killmask to all true
    std::fill(plan->killmask.begin(), plan->killmask.end(), (dedisp_bool)true);
    for (ii = 0; ii < plan->device_count; ii++)
      {
	err = dedisp_set_device(plan->gpuid);
	if (err != DEDISP_NO_ERROR)
	  {
	    dedisp_destroy_plan(plan);
	    return err;
	  }
	thrust::fill(plan->d_killmask[ii].begin(), plan->d_killmask[ii].end(),
		     (dedisp_bool)true);
      }
  }
  return DEDISP_NO_ERROR;
}
/*
dedisp_plan dedisp_set_stream(dedisp_plan plan, StreamType stream)
{
	plan->stream = stream;
	return plan;
}
*/

// Getters
// -------
dedisp_size         dedisp_get_max_delay(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	if( 0 == plan->dm_count ) { throw_getter_error(DEDISP_NO_DM_LIST_SET,0); }
	return plan->max_delay;
}
dedisp_size         dedisp_get_channel_count(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->nchans;
}
dedisp_size         dedisp_get_dm_count(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->dm_count;
}
const dedisp_float* dedisp_get_dm_list(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	if( 0 == plan->dm_count ) { throw_getter_error(DEDISP_NO_DM_LIST_SET,0); }
	return &plan->dm_list[0];
}
const dedisp_bool*  dedisp_get_killmask(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return &plan->killmask[0];
}
dedisp_float        dedisp_get_dt(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->dt;
}
dedisp_float        dedisp_get_f0(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->f0;
}
dedisp_float        dedisp_get_df(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	return plan->df;
}

// Warning: Big mother function
dedisp_error dedisp_execute_guru(const dedisp_plan  plan,
                                 dedisp_size        nsamps,
                                 const dedisp_byte* in,
                                 dedisp_size        in_nbits,
                                 dedisp_size        in_stride,
                                 dedisp_byte*       out,
                                 dedisp_size        out_nbits,
                                 dedisp_size        out_stride,
                                 dedisp_size        first_dm_idx,
                                 dedisp_size        dm_count,
				 unsigned           flags)
{

        int device_idx;
        cudaGetDevice(&device_idx);

	enum {
		BITS_PER_BYTE  = 8,
		BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
	};

	dedisp_size out_bytes_per_sample = out_nbits / (sizeof(dedisp_byte) *
	                                                BITS_PER_BYTE);

	if( 0 == in || 0 == out ) {
		throw_error(DEDISP_INVALID_POINTER);
	}
	// Note: Must be careful with integer division
	if( in_stride < plan->nchans*in_nbits/(sizeof(dedisp_byte)*BITS_PER_BYTE) ||
	    out_stride < (nsamps - plan->max_delay)*out_bytes_per_sample ) {
		throw_error(DEDISP_INVALID_STRIDE);
	}
	if( 0 == plan->dm_count ) {
		throw_error(DEDISP_NO_DM_LIST_SET);
	}
	if( nsamps < plan->max_delay ) {
		throw_error(DEDISP_TOO_FEW_NSAMPS);
	}

	// Check for valid synchronisation flags
	if( flags & DEDISP_ASYNC && flags & DEDISP_WAIT ) {
		throw_error(DEDISP_INVALID_FLAG_COMBINATION);
	}

	// Check for valid nbits values
	if( in_nbits  != 1 &&
	    in_nbits  != 2 &&
	    in_nbits  != 4 &&
	    in_nbits  != 8 &&
	    in_nbits  != 16 &&
	    in_nbits  != 32 ) {
		throw_error(DEDISP_UNSUPPORTED_IN_NBITS);
	}
	if( out_nbits != 8 &&
	    out_nbits != 16 &&
	    out_nbits != 32 ) {
		throw_error(DEDISP_UNSUPPORTED_OUT_NBITS);
	}

    // will never use the host memory
    bool using_host_memory = false;
	// Copy the lookup tables to constant memory on the device
	// TODO: This was much tidier, but thanks to CUDA's insistence on
	//         breaking its API in v5.0 I had to mess it up like this.
	// NEW: dereferenced d_delay_table to point to pointers on current device
    // Code below will be moved to the DedispPlan.hpp, so that this data is copied only once
	cudaMemcpyToSymbolAsync(c_delay_table,
	                        thrust::raw_pointer_cast(&plan->d_delay_table[device_idx][0]),
							plan->nchans * sizeof(dedisp_float),
							0, cudaMemcpyDeviceToDevice, 0);
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		throw_error(DEDISP_MEM_COPY_FAILED);
	}
	cudaMemcpyToSymbolAsync(c_killmask,
	                        thrust::raw_pointer_cast(&plan->d_killmask[device_idx][0]),
							plan->nchans * sizeof(dedisp_bool),
							0, cudaMemcpyDeviceToDevice, 0);
	cudaThreadSynchronize();
	error = cudaGetLastError();
	if( error != cudaSuccess ) {
		throw_error(DEDISP_MEM_COPY_FAILED);
	}

	// Compute the problem decomposition
	dedisp_size nsamps_computed = nsamps - plan->max_delay;
	// Specify the maximum gulp size
	dedisp_size nsamps_computed_gulp_max;
	nsamps_computed_gulp_max = nsamps_computed;

	// Compute derived counts for maximum gulp size
	dedisp_size chans_per_word  = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;
	dedisp_size nchan_words     = plan->nchans / chans_per_word;

	// We use words for processing but allow arbitrary byte strides, which are
	//   not necessarily friendly.
    // Our stride is friendly like a fluffy bunny
	bool friendly_in_stride = (0 == in_stride % BYTES_PER_WORD);

	// Note: If desired, this could be rounded up, e.g., to a power of 2
	dedisp_size in_buf_stride_words      = nchan_words;
	dedisp_size in_count_gulp_max        = nsamps * in_buf_stride_words;

	dedisp_size nsamps_padded_gulp_max   = div_round_up(nsamps_computed_gulp_max,
	                                                    DEDISP_SAMPS_PER_THREAD)
		* DEDISP_SAMPS_PER_THREAD + plan->max_delay;
	dedisp_size in_count_padded_gulp_max =
		nsamps_padded_gulp_max * in_buf_stride_words;

	// TODO: Make this a parameter?
	dedisp_size min_in_nbits = 0;
	dedisp_size unpacked_in_nbits = in_nbits;
	dedisp_size unpacked_chans_per_word =
		sizeof(dedisp_word)*BITS_PER_BYTE / unpacked_in_nbits;
	dedisp_size unpacked_nchan_words = plan->nchans / unpacked_chans_per_word;
	dedisp_size unpacked_buf_stride_words = unpacked_nchan_words;
	dedisp_size unpacked_count_padded_gulp_max =
		nsamps_padded_gulp_max * unpacked_buf_stride_words;

	dedisp_size out_stride_gulp_samples  = nsamps_computed_gulp_max;
	dedisp_size out_stride_gulp_bytes    =
		out_stride_gulp_samples * out_bytes_per_sample;
	dedisp_size out_count_gulp_max       = out_stride_gulp_bytes * dm_count;

	// Organise device memory pointers
	// -------------------------------
	const dedisp_word* d_in = 0;
	dedisp_word*       d_transposed = 0;
	dedisp_word*       d_unpacked = 0;
	dedisp_byte*       d_out = 0;
	thrust::device_vector<dedisp_word> d_in_buf;
	thrust::device_vector<dedisp_word> d_transposed_buf;
	thrust::device_vector<dedisp_word> d_unpacked_buf;
	thrust::device_vector<dedisp_byte> d_out_buf;

	d_in = (dedisp_word*)in;
	d_out = out;

	//// Note: * 2 here is for the time-scrunched copies of the data
	try { d_transposed_buf.resize(in_count_padded_gulp_max/* * 2 */); }
	catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	d_transposed = thrust::raw_pointer_cast(&d_transposed_buf[0]);

	// Note: * 2 here is for the time-scrunched copies of the data
	try { d_unpacked_buf.resize(unpacked_count_padded_gulp_max * 2); }
	catch(...) { throw_error(DEDISP_MEM_ALLOC_FAILED); }
	d_unpacked = thrust::raw_pointer_cast(&d_unpacked_buf[0]);
	// -------------------------------

	// The stride (in words) between differently-scrunched copies of the
	//   unpacked data.
	dedisp_size scrunch_stride = unpacked_count_padded_gulp_max;

	cudaStream_t stream = plan->stream;

	// Gulp loop
	for( dedisp_size gulp_samp_idx=0;
	     gulp_samp_idx<nsamps_computed;
	     gulp_samp_idx+=nsamps_computed_gulp_max ) {

		dedisp_size nsamps_computed_gulp = min(nsamps_computed_gulp_max,
		                                       nsamps_computed-gulp_samp_idx);
		dedisp_size nsamps_gulp          = nsamps_computed_gulp + plan->max_delay;
		dedisp_size nsamps_padded_gulp   = div_round_up(nsamps_computed_gulp,
		                                                DEDISP_SAMPS_PER_THREAD)
			* DEDISP_SAMPS_PER_THREAD + plan->max_delay;

		// Transpose the words in the input
		Transpose<dedisp_word> transpose;
		transpose.transpose(d_in,
		                    nchan_words, nsamps_gulp,
		                    in_buf_stride_words, nsamps_padded_gulp,
		                    d_transposed);

		// Unpack the transposed data
		unpack(d_transposed, nsamps_padded_gulp, nchan_words,
		       d_unpacked,
		       in_nbits, unpacked_in_nbits);

		if( !dedisperse(//d_transposed,
		                d_unpacked,
		                nsamps_padded_gulp,
		                nsamps_computed_gulp,
		                unpacked_in_nbits, //in_nbits,
		                plan->nchans,
		                1,
		                thrust::raw_pointer_cast(&plan->d_dm_list[device_idx][first_dm_idx]),
		                dm_count,
		                1,
		                d_out,
		                out_stride_gulp_samples,
		                out_nbits,
		                1, 0, 0, 0, 0) ) {
			throw_error(DEDISP_INTERNAL_GPU_ERROR);
		}

	} // End of gulp loop

	if( !(flags & DEDISP_ASYNC) ) {
		cudaStreamSynchronize(stream);
	}

	// Phew!
	return DEDISP_NO_ERROR;
}

dedisp_error dedisp_execute_adv(const dedisp_plan  plan,
                                dedisp_size        nsamps,
                                const dedisp_byte* in,
                                dedisp_size        in_nbits,
                                dedisp_size        in_stride,
                                dedisp_byte*       out,
                                dedisp_size        out_nbits,
                                dedisp_size        out_stride,
                                unsigned           flags)
{
	dedisp_size first_dm_idx = 0;
	dedisp_size dm_count = plan->dm_count;
	return dedisp_execute_guru(plan, nsamps,
	                           in, in_nbits, in_stride,
	                           out, out_nbits, out_stride,
	                           first_dm_idx, dm_count,
	                           flags);
}

dedisp_error dedisp_execute(const dedisp_plan  plan,
                            dedisp_size        nsamps,
                            const dedisp_byte* in,
                            dedisp_size        in_nbits,
                            dedisp_byte*       out,
                            dedisp_size        out_nbits,
                            unsigned           flags)
{

	enum {
		BITS_PER_BYTE = 8
	};

	dedisp_error retval;

       	// Note: The default out_stride is nsamps - plan->max_delay
	dedisp_size out_bytes_per_sample =
	  out_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);

	// Note: Must be careful with integer division
	dedisp_size in_stride =
	  plan->nchans * in_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);
	dedisp_size out_stride = (nsamps - plan->max_delay) * out_bytes_per_sample;

    dedisp_size first_dm_idx = 0;
    dedisp_size dm_count = plan->dm_count;

    return dedisp_execute_guru(plan, nsamps,
	                           in, in_nbits, in_stride,
	                           out, out_nbits, out_stride,
	                           first_dm_idx, dm_count,
	                           flags);
}


dedisp_error dedisp_sync(void)
{
	if( cudaThreadSynchronize() != cudaSuccess )
		throw_error(DEDISP_PRIOR_GPU_ERROR);
	else
		return DEDISP_NO_ERROR;
}

void dedisp_destroy_plan(dedisp_plan plan)
{
	if( plan ) {
		delete plan;
	}
}

const char* dedisp_get_error_string(dedisp_error error)
{
	switch( error ) {
	case DEDISP_NO_ERROR:
		return "No error";
	case DEDISP_MEM_ALLOC_FAILED:
		return "Memory allocation failed";
	case DEDISP_MEM_COPY_FAILED:
		return "Memory copy failed";
	case DEDISP_INVALID_DEVICE_INDEX:
		return "Invalid device index";
	case DEDISP_DEVICE_ALREADY_SET:
		return "Device is already set and cannot be changed";
	case DEDISP_NCHANS_EXCEEDS_LIMIT:
		return "No. channels exceeds internal limit";
	case DEDISP_INVALID_PLAN:
		return "Invalid plan";
	case DEDISP_INVALID_POINTER:
		return "Invalid pointer";
	case DEDISP_INVALID_STRIDE:
		return "Invalid stride";
	case DEDISP_NO_DM_LIST_SET:
		return "No DM list has been set";
	case DEDISP_TOO_FEW_NSAMPS:
		return "No. samples < maximum delay";
	case DEDISP_INVALID_FLAG_COMBINATION:
		return "Invalid flag combination";
	case DEDISP_UNSUPPORTED_IN_NBITS:
		return "Unsupported in_nbits value";
	case DEDISP_UNSUPPORTED_OUT_NBITS:
		return "Unsupported out_nbits value";
	case DEDISP_PRIOR_GPU_ERROR:
		return "Prior GPU error.";
	case DEDISP_INTERNAL_GPU_ERROR:
		return "Internal GPU error. Please contact the author(s).";
	case DEDISP_UNKNOWN_ERROR:
		return "Unknown error. Please contact the author(s).";
	default:
		return "Invalid error code";
	}
}

dedisp_error dedisp_enable_adaptive_dt(dedisp_plan  plan,
                                       dedisp_float pulse_width,
                                       dedisp_float tol)
{
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	plan->scrunching_enabled = true;
	plan->pulse_width = pulse_width;
	plan->scrunch_tol = tol;
	return update_scrunch_list(plan);
}
dedisp_error dedisp_disable_adaptive_dt(dedisp_plan plan) {
	if( !plan ) { throw_error(DEDISP_INVALID_PLAN); }
	plan->scrunching_enabled = false;
	return update_scrunch_list(plan);
}
dedisp_bool dedisp_using_adaptive_dt(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,false); }
	return plan->scrunching_enabled;
}
const dedisp_size* dedisp_get_dt_factors(const dedisp_plan plan) {
	if( !plan ) { throw_getter_error(DEDISP_INVALID_PLAN,0); }
	if( 0 == plan->dm_count ) { throw_getter_error(DEDISP_NO_DM_LIST_SET,0); }
	return &plan->scrunch_list[0];
}
// ----------------
