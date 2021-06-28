#include "includes.h"
__global__ void reduce_max_kernel(float *d_out, const float *d_logLum, int size) {

int tid         = threadIdx.x;                              // Local thread index
int myId        = blockIdx.x * blockDim.x + threadIdx.x;    // Global thread index

extern __shared__ float temp[];

// --- Loading data to shared memory. All the threads contribute to loading the data to shared memory.
temp[tid] = (myId < size) ? d_logLum[myId] : -10000000;

// --- Your solution
// if (myId < size) { temp[tid] = d_logLum[myId]; } else { temp[tid] = d_logLum[tid]; }

// --- Before going further, we have to make sure that all the shared memory loads have been completed
__syncthreads();

// --- Reduction in shared memory. Only half of the threads contribute to reduction.
for (unsigned int s=blockDim.x/2; s>0; s>>=1)
{
if (tid < s) { temp[tid] = fmaxf(temp[tid], temp[tid + s]); }
// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
__syncthreads();
}

// --- Your solution
//for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//    if (tid < s) { if (myId < size) { temp[tid] = fmaxf(d_logLum[myId + s], d_logLum[myId]); } else { temp[tid] = d_logLum[tid]; } }
//    __syncthreads();
//}

if (tid == 0) {
d_out[blockIdx.x] = temp[0];
}
}