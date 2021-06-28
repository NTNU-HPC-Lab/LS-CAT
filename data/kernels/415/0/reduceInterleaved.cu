#include "includes.h"

/*
* This code implements the interleaved Pair approaches to
* parallel reduction in CUDA. For this example, the sum operation is used.
*/

// Recursive Implementation of Interleaved Pair Approach
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
{
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

// boundary check
if(idx >= n) return;

// in-place reduction in global memory
for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
{
if (tid < stride)
{
g_idata[idx] += g_idata[idx + stride];
}

__syncthreads();
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = g_idata[idx];
}