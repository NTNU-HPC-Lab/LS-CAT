#include "includes.h"
__global__ void pi_optimized(float* x, float* y, int* global_count) {
__shared__ int counts[nthreads];

//int globalId = blockIdx.x * blockDim.x + nitemsperthread * threadIdx.x;
int globalId = blockIdx.x * blockDim.x + threadIdx.x;

int thread_count=0;
for (int i=0; i<nitemsperthread; i++) {
int idx = globalId+(i*nthreads*nblocks);
if (idx < nsamples) {
if (x[idx]*x[idx] + y[idx]*y[idx] < 1.0) {
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