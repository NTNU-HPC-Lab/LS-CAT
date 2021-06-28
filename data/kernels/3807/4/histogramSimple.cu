#include "includes.h"
__global__ void histogramSimple(int* d_out, const int* d_in, const int BINS_COUNT) {
int tid = threadIdx.x + blockDim.x * blockIdx.x;
atomicAdd(&(d_out[d_in[tid] % BINS_COUNT]), 1);
}