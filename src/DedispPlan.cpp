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
  This is a simple C++ wrapper class for the dedisp library
*/

#include <stdexcept>
#include <string>
#include <sstream>

#include <dedisp/dedisp.hpp>
#include <dedisp/DedispPlan.hpp>


// Constructor
DedispPlan::DedispPlan(size_type  nchans,
                       float_type dt,
                       float_type f0,
                       float_type df,
                       int gpuid) {
	check_error( dedisp_create_plan(&m_plan,
	                                nchans,
	                                dt, f0, df, gpuid),
	             "dedisp_create_plan" );
}
// Destructor
DedispPlan::~DedispPlan() {
	dedisp_destroy_plan(m_plan);
}

// Public interface
void DedispPlan::set_device(int device_idx) {
	check_error( dedisp_set_device(device_idx), "dedisp_set_device" );
}
void DedispPlan::set_gulp_size(size_type gulp_size) {
	check_error( dedisp_set_gulp_size(m_plan, gulp_size), "dedisp_set_gulp_size" );
}
void DedispPlan::set_killmask(const bool_type* killmask) {
	check_error( dedisp_set_killmask(m_plan, killmask), "dedisp_set_killmask" );
}
void DedispPlan::set_dm_list(const float_type* dm_list,
							 size_type         count) {
	check_error( dedisp_set_dm_list(m_plan, dm_list, count), "dedisp_set_dm_list" );
}
void DedispPlan::generate_dm_list(float_type dm_start,
                                  float_type dm_end,
                                  float_type ti,
                                  float_type tol) {
	check_error( dedisp_generate_dm_list(m_plan,
	                                     dm_start,
	                                     dm_end,
	                                     ti,
	                                     tol),
	             "dedisp_generate_dm_list" );
}

DedispPlan::size_type         DedispPlan::get_gulp_size()     const { return dedisp_get_gulp_size(m_plan); }
DedispPlan::float_type        DedispPlan::get_max_delay()     const { return dedisp_get_max_delay(m_plan); }
DedispPlan::size_type         DedispPlan::get_channel_count() const { return dedisp_get_channel_count(m_plan); }
DedispPlan::size_type         DedispPlan::get_dm_count()      const { return dedisp_get_dm_count(m_plan); }
const DedispPlan::float_type* DedispPlan::get_dm_list()       const { return dedisp_get_dm_list(m_plan); }
const DedispPlan::bool_type*  DedispPlan::get_killmask()      const { return dedisp_get_killmask(m_plan); }
DedispPlan::float_type        DedispPlan::get_dt()            const { return dedisp_get_dt(m_plan); }
DedispPlan::float_type        DedispPlan::get_df()            const { return dedisp_get_df(m_plan); }
DedispPlan::float_type        DedispPlan::get_f0()            const { return dedisp_get_f0(m_plan); }

void DedispPlan::execute(size_type        nsamps,
                         const byte_type* in,
                         size_type        in_nbits,
                         byte_type*       out,
                         size_type        out_nbits,
                         unsigned         flags) {
	check_error( dedisp_execute(m_plan,
	                            nsamps,
	                            in,
	                            in_nbits,
	                            out,
	                            out_nbits,
	                            flags),
	             "dedisp_execute" );
}
void DedispPlan::execute_adv(size_type        nsamps,
                             const byte_type* in,
                             size_type        in_nbits,
                             size_type        in_stride,
                             byte_type*       out,
                             size_type        out_nbits,
                             size_type        out_stride,
                             unsigned         flags) {
	check_error( dedisp_execute_adv(m_plan,
	                                nsamps,
	                                in,
	                                in_nbits,
	                                in_stride,
	                                out,
	                                out_nbits,
	                                out_stride,
	                                flags),
	             "dedisp_execute_adv" );
}
void DedispPlan::execute_guru(size_type        nsamps,
                              const byte_type* in,
                              size_type        in_nbits,
                              size_type        in_stride,
                              byte_type*       out,
                              size_type        out_nbits,
                              size_type        out_stride,
                              size_type        first_dm_idx,
                              size_type        dm_count,
                              unsigned         flags) {
	check_error( dedisp_execute_guru(m_plan,
	                                 nsamps,
	                                 in,
	                                 in_nbits,
	                                 in_stride,
	                                 out,
	                                 out_nbits,
	                                 out_stride,
	                                 first_dm_idx,
	                                 dm_count,
	                                 flags),
	             "dedisp_execute_guru" );
}
