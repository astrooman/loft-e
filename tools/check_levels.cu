#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <cuda.h>

#include "errors.hpp"

using std::cout;
using std::endl;

__global__ void transpose(float* __restrict__ in, float* __restrict__ out, unsigned int nchans, unsigned int ntimes) {

    // very horrible implementation or matrix transpose
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * ntimes;
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        out[start + tsamp] = in[idx + tsamp * nchans];
    }
}

__global__ void scale_factors(float* in, float **means, float **stdevs, unsigned int nchans, unsigned int ntimes, int param) {
    // calculates mean and standard deviation in every channel
    // assumes the data has been transposed

    // for now have one thread per frequency channel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float mean;
    float variance;

    float ntrec = 1.0f / (float)ntimes;
    float ntrec1 = 1.0f / (float)(ntimes - 1.0f);

    unsigned int start = idx * ntimes;
    mean = 0.0f;
    variance = 0.0f;
    // two-pass solution for now
    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        mean += in[start + tsamp] * ntrec;
    }
    means[param][idx] = mean;

    for (int tsamp = 0; tsamp < ntimes; tsamp++) {
        variance += (in[start + tsamp] - mean) * (in[start + tsamp] - mean);
    }
    variance *= ntrec1;
    // reciprocal of standard deviation
    stdevs[param][idx] = rsqrtf(variance) + 2.0 * (float)param;
}

int main(int argc, char *argv[])
{
        std::stringstream oss;
        int time_samp{0};
        time_samp = atoi(argv[2]);
        std::string inname, outname;
        cout << "Reading " <<  time_samp << " time samples from file: " << inname << endl;
        std::ifstream input_file(argv[1], std::ios_base::in | std::ios_base::binary);

        std::string read_param;
        char field[60];

        int strlen;

        input_file.seekg(369, input_file.beg);

        char *head = new char[4];

        input_file.read(head, 4);

        for (int ii = 0; ii < 4; ii++)
                cout << head[ii] << " ";

        cout << endl << endl;
        cout.flush();
        size_t to_read = 567 * time_samp * 4;

        float *data = new float[567 * time_samp];
        float *scaled = new float[567 * time_samp];
        float **means = new float*[4];
        float **stdevs = new float*[4];
        for (int ii = 0; ii < 4; ii++) {
            means[ii] = new float[567];
            stdevs[ii] = new float[567];
        }
        cout << "Reading some data now..." << endl;
        input_file.read(reinterpret_cast<char*>(data), to_read);
        input_file.close();

        for (int ii = 0; ii < 21; ii++)
                cout << data[ii] << " ";
        cout << endl << endl;

        float *d_data;
        float *d_trans;
        float *d_scaled;
        float **d_means;
        float **d_stdevs;
        float **h_means = new float*[4];
        float **h_stdevs = new float*[4];
        cudaCheckError(cudaMalloc((void**)&d_data, 567 * time_samp * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_trans, 567 * time_samp * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_scaled, 567 * time_samp * sizeof(float)));
        cudaCheckError(cudaMalloc((void**)&d_means, 4 * sizeof(float*)));
        cudaCheckError(cudaMalloc((void**)&d_stdevs, 4 * sizeof(float*)));
        for (int ii = 0; ii < 4; ii++) {
            cudaCheckError(cudaMalloc((void**)&h_means[ii], 567 * sizeof(float)));
            cudaMemset(h_means[ii], 0, 567 * sizeof(float));
            cudaCheckError(cudaMalloc((void**)&h_stdevs[ii], 567 * sizeof(float)));
            cudaMemset(h_stdevs[ii], 0, 567 * sizeof(float));
        }

        cudaCheckError(cudaMemcpy(d_means, h_means, 4 * sizeof(float*), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(d_stdevs, h_stdevs, 4 * sizeof(float*), cudaMemcpyHostToDevice));

        cudaCheckError(cudaMemcpy(d_data, data, 567 * time_samp * sizeof(float), cudaMemcpyHostToDevice));
        cout << "Starting the kernel...\n";

        for (int ii = 0; ii < 4; ii++) {
        //    transpose<<<1,567,0,0>>>(d_data, d_trans, 567, time_samp);
        //    scale_factors<<<1,567,0,0>>>(d_trans, d_means, d_stdevs, 567, time_samp, ii);
        }

        cudaDeviceSynchronize();

        cout << "Copying the results back...\n";

        for (int ii = 0; ii < 4; ii++) {
            cudaCheckError(cudaMemcpy(means[ii], h_means[ii], 567 * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheckError(cudaMemcpy(stdevs[ii], h_stdevs[ii], 567 * sizeof(float), cudaMemcpyDeviceToHost));
        }
        cout << "CHANNEL MEAN STDEV\n";
        for (int ii = 0; ii < 4; ii++) {
            cout << "Printing part " << ii << endl;
            for (int jj = 0; jj < 16; jj++)
                cout << jj << " " << means[ii][jj] << " " << stdevs[ii][jj] << endl;
        }

        for (int ii = 0; ii < 4; ii++) {
            cudaCheckError(cudaFree(h_means[ii]));
            cudaCheckError(cudaFree(h_stdevs[ii]));
            delete [] means[ii];
            delete [] stdevs[ii];
        }

        delete [] data;
        delete [] scaled;
        delete [] means;
        delete [] stdevs;
        delete [] h_means;
        delete [] h_stdevs;
        cudaCheckError(cudaFree(d_data));
        cudaCheckError(cudaFree(d_trans));
        cudaCheckError(cudaFree(d_scaled));
        cudaCheckError(cudaFree(d_means));
        cudaCheckError(cudaFree(d_stdevs));

        cudaDeviceReset();

        return 0;
}

