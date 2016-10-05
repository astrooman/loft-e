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

#ifndef _H_PAFRB_DEDISP_PLAN
#define _H_PAFRB_DEDISP_PLAN

#include <stdexcept>
#include <string>
#include <sstream>

#include <dedisp/dedisp.hpp>

class DedispPlan {
	// Private members
	dedisp_plan m_plan;

        // No copying or assignment
	DedispPlan(const DedispPlan& other);
	DedispPlan& operator=(const DedispPlan& other);
public:
	// Public types
	typedef dedisp_size  size_type;
	typedef dedisp_byte  byte_type;
	typedef dedisp_float float_type;
	typedef dedisp_bool  bool_type;

        // Constructor
	DedispPlan(size_type  nchans,
	           float_type dt,
	           float_type f0,
	           float_type df,
		       int gpuid);

	// Destructor
	~DedispPlan();

	static void set_device(int device_idx);

	// Public interface
	void set_gulp_size(size_type gulp_size);
	void set_killmask(const bool_type* killmask);
	void set_dm_list(const float_type* dm_list,
	                 size_type         count);
	void generate_dm_list(float_type dm_start,
	                      float_type dm_end,
	                      float_type ti,
	                      float_type tol);
	size_type         get_gulp_size()     const;
	float_type        get_max_delay()     const;
	size_type         get_channel_count() const;
	size_type         get_dm_count()      const;
	const float_type* get_dm_list()       const;
	const bool_type*  get_killmask()      const;
	float_type        get_dt()            const;
	float_type        get_df()            const;
	float_type        get_f0()            const;
	void execute(size_type        nsamps,
	             const byte_type* in,
	             size_type        in_nbits,
	             byte_type*       out,
	             size_type        out_nbits,
	             unsigned         flags);
	void execute_adv(size_type        nsamps,
	                 const byte_type* in,
	                 size_type        in_nbits,
	                 size_type        in_stride,
	                 byte_type*       out,
	                 size_type        out_nbits,
	                 size_type        out_stride,
	                 unsigned         flags);
	void execute_guru(size_type        nsamps,
	                  const byte_type* in,
	                  size_type        in_nbits,
	                  size_type        in_stride,
	                  byte_type*       out,
	                  size_type        out_nbits,
	                  size_type        out_stride,
	                  dedisp_size      first_dm_idx,
	                  dedisp_size      dm_count,
	                  unsigned         flags);
private:
	static void check_error(dedisp_error error, std::string function_name) {
		if( error != DEDISP_NO_ERROR ) {
			std::stringstream ss;
			ss << function_name << " failed: "
			   << dedisp_get_error_string(error)
			   << "\n";
			throw std::runtime_error(ss.str());
		}
	}
};

#endif
