#ifndef _H_PAFRB_FILTERBANK
#define _H_PAFRB_FILTERBANK

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

struct header_f
{

    std::string raw_file;
    std::string source_name;

    double az;                      // azimuth angle in deg
    double dec;                     // source declination
    double fch1;                    // frequency of the top channel in MHz
    double foff;                    // channel bandwidth in MHz
    double ra;                      // source right ascension
    double rdm;                     // reference DM
    double tsamp;                   // sampling time in seconds
    double tstart;                  // observation start time in MJD format
    double za;                      // zenith angle in deg

    int data_type;                  // data type ID
    int ibeam;                      // beam number
    int machine_id;
    int nbeams;
    int nbits;
    int nchans;
    int nifs;                       // something, something, something, DAAARK SIIIDEEEE
    int telescope_id;

};
//! Function that saves the filterbank file to the disk
/*!
    \param **ph_filterbank pointer to the host data
    \param nsamsp number of time samples to save
    \param head structure with all the information require for the filterbank header
    \param saved number of the filterbank files saved so far
*/
template<class FilType>
inline void SaveFilterbank(unsigned char **ph_filterbank, size_t nsamps, size_t start, header_f head, int stokes, int saved, std::string telescope, std::string outdir)
{

    std::ostringstream oss;
    std::string filename;

    int length{0};
    char field[60];
    char stokesid[4] = {'I', 'Q', 'U', 'V'};

    for (int ii = 0; ii < stokes; ii++) {
        oss.str("");
        oss << stokesid[ii] << "_" << std::setprecision(8) << std::fixed << head.tstart << "_" << telescope;
        filename = outdir + "/" + oss.str() + ".fil";
        std::fstream outfile(filename.c_str(), std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);

        // save only when the stream has been opened correctly
        if(outfile) {

            length = 12;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "HEADER_START");
            outfile.write(field, length * sizeof(char));

            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "telescope_id");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.telescope_id, sizeof(int));

            length = 11;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "rawdatafile");
            outfile.write(field, length * sizeof(char));
            length = head.raw_file.size();
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, head.raw_file.c_str());
            outfile.write(field, length * sizeof(char));

            length = 11;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "source_name");
            outfile.write(field, length * sizeof(char));
            length = head.source_name.size();
            strcpy(field, head.source_name.c_str());
            outfile.write((char*)&length, sizeof(int));
            outfile.write(field, length * sizeof(char));

            length = 10;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "machine_id");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.machine_id, sizeof(int));

            length = 9;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "data_type");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.data_type, sizeof(int));

            length = 8;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "az_start");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.az, sizeof(double));

            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "za_start");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.za, sizeof(double));

            length = 7;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "src_raj");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.ra, sizeof(double));

            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "src_dej");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.dec, sizeof(double));

            length = 6;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "tstart");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.tstart, sizeof(double));

            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "nchans");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.nchans, sizeof(int));

            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "nbeams");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.nbeams, sizeof(int));

            length = 5;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "tsamp");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.tsamp, sizeof(double));

            // bits per time sample
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "nbits");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.nbits, sizeof(int));

            // reference dm - not really sure what it does
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "refdm");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.rdm, sizeof(double));

            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "ibeam");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.ibeam, sizeof(int));

            length = 4;
            // the frequency of the top channel
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "fch1");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.fch1, sizeof(double));

            // channel bandwidth
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "foff");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.foff, sizeof(double));

            // number of if channels
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "nifs");
            outfile.write(field, length * sizeof(char));
            outfile.write((char*)&head.nifs, sizeof(int));

            length = 10;
            outfile.write((char*)&length, sizeof(int));
            strcpy(field, "HEADER_END");
            outfile.write(field, length * sizeof(char));

            size_t to_save = nsamps * head.nchans * head.nbits / 8;
            T *ph_filsave = ph_filterbank[ii];
            outfile.write(reinterpret_cast<char*>(&ph_filsave[start]), to_save);
            //outfile.write(reinterpret_cast<char*>(&ph_filterbank[start + ii * nsamps * head.nchans]), to_save);

        } else {
            std::cerr << "Problems with saving the filterbank file" << std::endl;
        }


        outfile.close();

    }

    std::cout << "Saved filterbank " << saved << " for beam " << head.ibeam << std::endl;

}

#endif
