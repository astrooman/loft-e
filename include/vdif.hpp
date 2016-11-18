#ifndef _H_PAFRB_PDIF
#define _H_PAFRB_PDIF

#include <fstream>
#include <ios>
#include <iostream>

#include <arpa/inet.h>
#include <endian.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>
#include <cufft.h>


#define HEADER 64   // header is 64 bytes long
#define BYTES_PER_WORD 8
#define WORDS_PER_PACKET 896

using std::cout;
using std::endl;

struct header_s {
    // thigs are listed in the order they appear in the vdif header
    unsigned int refs;      // seconds from reference epoch
    unsigned int frameno;   // data frame within the current period
    unsigned int epoch;      // reference epoch
    unsigned int framelen;    // data array length in units of 8 bytes
    unsigned int nchans;      // number of channels - 1 (why the hell minus 1?!)
    unsigned int station;    // station ID
    unsigned int thread;     // thread ID
    unsigned int inbits;    // bits per sample

};

// should read header from the data packet
inline void get_header(unsigned char* packet, header_s &head)
{

    long long *hword = new long long[8];
    // stuff arrives in the network order and has to be changed into host order
    for (int ii = 0; ii < 8; ii++) {
	hword[ii] = be64toh(*(reinterpret_cast<long long*>(packet+ii*8)));
    }

    head.refs = (unsigned int)(hword[0] & 0x3fffffff);
    head.frameno = (unsigned int)(hword[1] & 0xffffff);
    head.epoch = (unsigned int)((hword[1] >> 24) & 0x3f);
    head.framelen = (unsigned int)(hword[2] & 0xffffff);
    head.nchans = (unsigned int)(1 << (hword[2] >> 24) & 0x1f);
    head.station = (char)(hword[3] & 0xff00) + (char)(hword[3] & 0xff);
    head.thread = (unsigned int)((hword[3] >> 16) & 0x3ff);
    head.inbits = (unsigned int)((hword[3] >> 26) & 0x1f); + 1;
}

#endif
