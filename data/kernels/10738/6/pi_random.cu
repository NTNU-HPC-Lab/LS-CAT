#include "includes.h"
__global__ void pi_random(float* x, float* y, int* global_count) {
__shared__ int counts[nthreads];

int globalId = blockIdx.x * blockDim.x + nitemsperthread * threadIdx.x;

int thread_count=0;
for (int i=0; i<nitemsperthread; i++) {
if (globalId+i < nsamples) {
if (x[globalId+i]*x[globalId+i] + y[globalId+i]*y[globalId+i] < 1.0) {
thread_count++;
}
}
}

counts[threadIdx.x] = thread_count;
__syncthreads();

if (threadIdx.x == 0) {
int block_count = 0;
for (int i=0; i<nthreads; i++) {
block_count += counts[i];
}
global_count[blockIdx.x] = block_count;
}
}