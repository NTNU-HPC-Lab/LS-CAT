#include "includes.h"
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
// set the thread id.
unsigned int tid = threadIdx.x;
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

// convert global data pointer to the local pointer of this block.
int *idata = g_idata + blockIdx.x * blockDim.x;

// boundary check.
if (idx >= n) return;

for (int stride = 1; stride < blockDim.x; stride *= 2)
{
// convert tid into local array index.
int index = 2 * stride * tid;

if (index < blockDim.x)
{
idata[index] += idata[index + stride];
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