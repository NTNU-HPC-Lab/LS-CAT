#include "includes.h"
__global__ void reduceUnrollWarps (int *g_idata, int *g_odata, unsigned int n)
{
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

// convert global data pointer to the local pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x * 2;

// unrolling 2
if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

__syncthreads();

// in-place reduction in global memory
for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
{
if (tid < stride)
{
idata[tid] += idata[tid + stride];
}

// synchronize within threadblock
__syncthreads();
}

// unrolling last warp
if (tid < 32)
{
volatile int *vsmem = idata;
vsmem[tid] += vsmem[tid + 32];
vsmem[tid] += vsmem[tid + 16];
vsmem[tid] += vsmem[tid +  8];
vsmem[tid] += vsmem[tid +  4];
vsmem[tid] += vsmem[tid +  2];
vsmem[tid] += vsmem[tid +  1];
}

if (tid == 0) g_odata[blockIdx.x] = idata[0];
}