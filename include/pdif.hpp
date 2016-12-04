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


#define HEADER 32   // header is 64 bytes long
#define BYTES_PER_WORD 8
#define WORDS_PER_PACKET 896

using std::cout;
using std::endl;

struct header_s {
    // thigs are listed in the order they appear in the pdif header
    // WORD 0
    int ref_s;      // seconds from reference epoch
    bool comp;      // complex data flag: real = 0, complex = 1
    bool invalid;   // invalid data: valid = 0, invalid = 1
    // WORD 1
    int frame_no;   // data frame within the current period
    // WORD 2
    int arr_len;    // data array length in units of 8 bytes
    int in_bits;    // bits per sample
    int version;    // pdif version number; 1 for PDIF2 (doesn't make much sense)
    // WORD 3
    int station;    // station ID
    int rep;        // sample representation: 0 - offset binary, 1 - 2's compliment, 2 - IEEE floating point
    unsigned int epoch;      // reference epoch
    // WORD 4
    unsigned int nchans;      // number of channels - 1 (why the hell minus 1?!)
    int block_len;  // sample block length - 1 in units of 8 bytes (must be large enough to hold
                    // at least one complete sample: sample size * nchans (* 2 for complex data))
    // WORD 5
    int group;      // group ID
    int thread;     // thread ID
    // WORD 6
    int period;     // in seconds - number  seconds during which there is exactly an integral number of sample periods
    // WORD 7 - reserved for future use
    // WORD 8
    int sipp_m;     // sample intervals per period (MSB)
    // WORD 9
    int sipp_l;     // sample intervals per period (LSB)
    // WORD 10
    int synch;      // 32-bit synchronisation word (‘0xadeadbee’), always present
    // WORD 11 - reserved for future use
    // WORD 12
    int edv;        // extended data version
    // WORD 12, 13, 14, 15 - extended user data
    unsigned long long sipp;
};

// should read header from the data packet
inline void get_header(unsigned char* packet, header_s &head)
{

    long long *hword = new long long[8];
    // stuff arrives in the network order and has to be changed into host order
    for (int ii = 0; ii < 8; ii++) {
	hword[ii] = be64toh(*(reinterpret_cast<long long*>(packet+ii*8)));
    }

    head.frame_no = (unsigned int)(hword[0] & 0xffffffff);
    head.ref_s = (unsigned int)((hword[0] >> 32) & 0xffffffff);
    head.comp = (bool)((hword[0] >> 62) & 0x1);
    head.invalid = (bool)((hword[0] >> 63)  & 0x1);

    head.station = (int)(hword[1] & 0xffff);
    head.rep = (int)((hword[1] >> 22) & 0xf);
    head.epoch = (unsigned int)((hword[1] >> 26) & 0x3f);
    head.arr_len = (int)((hword[1] >> 32) & 0xffffff) + 1;
    head.in_bits = (int)((hword[1] >> 56) & 0x1f);
    head.version = (int)((hword[1] >> 61) & 0x7) + 1;

    head.group = (int)(hword[2] & 0xffff);
    head.thread = (int)((hword[2] >> 16) & 0xffff);
    head.nchans = ((unsigned int)((hword[2] >> 32) & 0xffff) + 1) / 2;
    head.block_len = (int)((hword[2] >> 48) & 0xffff) + 1;

    head.period = (int)((hword[3] >> 32) & 0xffff);

    head.sipp = (unsigned long long)(hword[4]);
//    char *hword0c = reinterpret_cast<char*>(&hword0);
//    char *hword1c = reinterpret_cast<char*>(&hword1);

//    head.frame_no = (unsigned int)(hwordc[0][0] | (hwordc[0][1] << 8) | (hwordc[0][2] << 16) | (hwordc[0][3] << 24));
    //head.ref_s = (unsigned int)(hwordc[0][4] | (hwordc[0][5] << 8) | (hwordc[0][6] << 16) | ((hwordc[0][7] << 24) & 0x3f000000));
//    head.comp = (bool)((hwordc[0][7] & 0x40) >> 6);
//    head.invalid = (bool)((hwordc[0][7] & 0x80) >> 7);

//    head.in_bits = (int)(hwordc[1][7] & 0x1f);         // 0x1f -> 00011111
//    head.version = (int)((hwordc[1][7] & 0xe0) >> 5);         // 0xe0 -> 11100000

//    head.ref_s = (int)(packet[0] | (packet[1] << 8) | (packet[2] << 16) | ((packet[3] << 24) & 0x3f000000));
//    head.comp = (bool)((packet[3] & 0x40) >> 6);
//    head.invalid = (bool)((packet[3] & 0x80) >> 7);
//    head.frame_no = (int)(packet[4] | (packet[5] << 8) | (packet[6] << 16) | (packet[7] << 24));
//    head.arr_len = (int)(packet[8] | (packet[9] << 8) | (packet[10] << 16));
//    head.in_bits = (int)(packet[11] & 0x1f);         // 0x1f -> 00011111
//    head.version = (int)((packet[11] & 0xe0) >> 5);         // 0xe0 -> 11100000
//    head.station = (int)(packet[12] | (packet[13] << 16));
//    head.rep = (int)((packet[14] >> 6 ) | ((packet[15] & 0x3) << 2));      // 0x3 -> 00000011
//    head.epoch = (int)((packet[15] & 0xfc) >> 2);     // 0xfc -> 11111100
//    head.nchans = (int)(packet[16] | (packet[17] << 8));
//    head.block_len = (int)(packet[18] | (packet[19] << 8));
//    head.group = (int)(packet[20] | (packet[21] << 8));
//    head.thread = (int)(packet[22] | (packet[23] << 8));
//    head.period = (int)(packet[24] | (packet[25] << 8));
//    head.synch = (int)(packet[40] | (packet[41] << 8) | (packet[42] << 16) | (packet[43] << 24));

    // cast to long, just in case compiler is a bit daft

    head.synch = (int)(hword[5] >> 32);

//    long tsipp = (long)(packet[32] | (packet[33] << 8) | (packet[34] << 16) | (packet[35] << 24)
//                    | ((long)packet[36] << 32) | ((long)packet[37] << 40) | ((long)packet[38] << 48) | ((long)packet[39] << 56));
//    head.sipp = tsipp;


//    cout << "PDIF version " << head.version << endl;
//    cout << "Seconds from the reference epoch: " << head.ref_s << endl;
//    cout << "Invalid data: " << head.invalid << endl;
//    cout << "Complex data: " << head.comp << endl;
//    cout << "In bits: " << head.in_bits << endl;
//    cout << "Station: " << std::hex << head.station << std::dec << endl;
//    cout << "Thread number: " << head.thread << endl;
//    cout << "Group number: " << head.group << endl;
//    cout << "Reference epoch: " << head.epoch << endl;
//    cout << "Data array length: " << head.arr_len << endl;
//    cout << "Sample block length: " << head.block_len << endl;
//    cout << "Period: " << head.period << endl;
//    cout << "Number of channels: " << head.nchans << endl;
//    cout << "Data frame within the current period: " << head.frame_no << endl;
//    cout << "Sample intervals per period: " << head.sipp << endl;
//    cout << "Synchronisation word: " << std::hex << head.synch << std::dec << endl;
    cout.flush();
}


/*void get_data(unsigned char *data, cufftComplex *pola, cufftComplex *polb, int &d_begin, int frame, int &previous_frame) {

    unsigned int idx = 0;
    unsigned int idx2 = 0;

    if((frame - previous_frame) > 1) {
        // count words only as one word provides one full time sample per polarisation
        d_begin += (frame - previous_frame) * 7 * 128;
	cout << "We missed " << frame - previous_frame << " frames!!" << endl;
	cout.flush();
    } else {
        d_begin += 7 * 128;
	cout << "Didn't miss anything!!" << endl;
	cout.flush();
    }

    int fpga_id = frame % 48;
    #pragma unroll
    for (int chan = 0; chan < 7; chan++) {
        for (int sample = 0; sample < 128; sample++) {
            idx = (sample * 7 + chan) * BYTES_PER_WORD;    // get the  start of the word in the received data array
            idx2 = chan * 128 + sample + fpga_id * WORDS_PER_PACKET;        // get the position in the buffer
            pola[idx2].x = (float)(data[HEADER + idx + 3] | (data[HEADER + idx + 2] << 8));
            pola[idx2].y = (float)(data[HEADER + idx + 1] | (data[HEADER + idx + 0] << 8));
            polb[idx2].x = (float)(data[HEADER + idx + 7] | (data[HEADER + idx + 6] << 8));
            polb[idx2].y = (float)(data[HEADER + idx + 5] | (data[HEADER + idx + 4] << 8));
        }
    }

    previous_frame = frame;

} */

#endif
