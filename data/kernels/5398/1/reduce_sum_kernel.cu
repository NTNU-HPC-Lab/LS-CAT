#include "includes.h"
__global__ void reduce_sum_kernel(const float *input, float *d_out,  int size) {
int tid         = threadIdx.x;                              // Local thread index
int myId        = blockIdx.x*(blockDim.x*2) + threadIdx.x;   // Global thread index

extern __shared__ float tempsum[]; //shared memory

// --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
tempsum[tid] = (myId < size) ? input[myId] + input[myId+blockDim.x] : 0.0f;

// --- make sure that all the shared memory loads have been completed
__syncthreads();

// --- Reduction in shared memory. Only half of the threads contribute to reduction.
for (unsigned int s=blockDim.x/2; s>0; s>>=1){
if (tid < s) { tempsum[tid] += tempsum[tid + s]; }
// --- make sure that all memory operations have been completed
__syncthreads();
}

if (tid == 0) {
d_out[blockIdx.x] = tempsum[0];
}
}