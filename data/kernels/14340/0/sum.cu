#include "includes.h"
/**
* This is an example from the paper "Formal Semantics of Heterogeneous CUDA-C:
* A Modular Approach with Applications" by Chris Hathhorn et al.
*/


#define N 8
#define NBLOCKS 4
#define NTHREADS (N/NBLOCKS)


__global__ void sum(int* in, int* out) {
extern __shared__ int shared[];
int i, tid = threadIdx.x,
bid = blockIdx.x,
bdim = blockDim.x;

shared[tid] = in[bid * bdim + tid];

__syncthreads();
if(tid < bdim/2) {
shared[tid] += shared[bdim/2 + tid];
}
__syncthreads();
if(tid == 0) {
for (i = 1; i != (bdim/2) + (bdim%2); ++i) {
shared[0] += shared[i];
}
out[bid] = shared[0];
}
}