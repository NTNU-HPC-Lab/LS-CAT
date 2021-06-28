#include "includes.h"
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n)
{
// set the thread id.
unsigned int tid = threadIdx.x;
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;

// convert global data pointer to the local pointer of this block.
int *idata = g_idata + blockIdx.x * blockDim.x * 2;
// unrolling 2 data blocks.
if (idx + blockDim.x < n)
{
g_idata[idx] += g_idata[idx + blockDim.x];
}
__syncthreads();

// boundary check.
if (idx >= n) return;

// in-place reduction in global memory
for (int stride = blockDim.x/2; stride>0; stride>>=1)
{
if (tid < stride)
{
idata[tid] += idata[tid + stride];
}

// synchronize within threadblock.
__syncthreads();
}

// write result for this block to global mem.
if (tid == 0)
{
g_odata[blockIdx.x] = idata[0];
}
}