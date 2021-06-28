#include "includes.h"
__global__ void entrySearch_max_int_kernel(int *g_iarr, int *g_maxarr, int size)
{
// create shared memory
extern __shared__ int sarr_int[];

// load shared mem
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

if(i + blockDim.x < size) {
if(g_iarr[i] > g_iarr[i + blockDim.x]) {
sarr_int[tid] = g_iarr[i];
} else {
sarr_int[tid] = g_iarr[i + blockDim.x];
}
} else if (i < size) {
sarr_int[tid] = g_iarr[i];
} else {
sarr_int[tid] = INT_MIN;
}

__syncthreads();

// do comparison in shared mem
for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
if(tid < s) {
if(sarr_int[tid] < sarr_int[tid + s]) {
sarr_int[tid] = sarr_int[tid + s];
}
}
__syncthreads();
}

// write result for this block to global mem
if(tid == 0) {
g_maxarr[blockIdx.x] = sarr_int[0];
}
}