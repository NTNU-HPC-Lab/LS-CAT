#include "includes.h"
__global__  void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
// set thread id.
unsigned int tid = threadIdx.x;
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

// convert global data pointer to th local pointer of this block.
int *idata = g_idata + blockIdx.x * blockDim.x;

// boundary check.
if (idx >= n) return;

// in-place reduction in global memory.
for (int stride = 1; stride < blockDim.x; stride *= 2)
{
if ( (tid % (2 * stride)) == 0)
{
idata[tid] += idata[tid + stride];
}

// synchronize within block.
__syncthreads();
}

// write result for this block to global mem.
if (tid == 0)
{
g_odata[blockIdx.x] = idata[0];
}
}