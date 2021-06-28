#include "includes.h"
__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n)
{
// set thread ID
unsigned int tid = threadIdx.x;
int *idata = g_idata + blockIdx.x * blockDim.x;

// boundary check
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx >= n) return;

// in-place reduction in global memory
if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

__syncthreads();

if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

__syncthreads();

if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

__syncthreads();

if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

__syncthreads();

// unrolling warp
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

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = idata[0];
}