#include "includes.h"
// In CUDA we trust.

// When compiling, use -std=c++11 or higher.



__global__ void histogramSimple(int* d_out, const int* d_in, const int BINS_COUNT) {
int tid = threadIdx.x + blockDim.x * blockIdx.x;
atomicAdd(&(d_out[d_in[tid] % BINS_COUNT]), 1);
}