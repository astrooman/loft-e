#ifndef _H_PAFRB_OBS_TIME
#define _H_PAFRB_OBS_TIME

struct ObsTime {

    unsigned int startepoch;            // reference epoch at the start of the observation
    unsigned int startsecond;           // seconds from the reference epoch at the start of the observation
    unsigned int framet;                 // frame number from the start of the observation

};

#endif
