#include "includes.h"
#define LOG 0

/*
* An implementation of parallel reduction using nested kernel launches from
* CUDA kernels. This version adds optimizations on to the work in
* nestedReduce.cu.
*/

// Recursive Implementation of Interleaved Pair Approach
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)
{
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

// convert global data pointer to the local pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x;

// boundary check
if (idx >= n) return;

// in-place reduction in global memory
for (int stride = 1; stride < blockDim.x; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
idata[tid] += idata[tid + stride];
}

// synchronize within threadblock
__syncthreads();
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = idata[0];
}