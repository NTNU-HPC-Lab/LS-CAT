#include "includes.h"
__global__ void reduceSmemDyn(int *g_idata, int *g_odata, unsigned int n)
{
extern __shared__ int smem[];

// set thread ID
unsigned int tid = threadIdx.x;
int *idata = g_idata + blockIdx.x * blockDim.x;

// set to smem by each threads
smem[tid] = idata[tid];
__syncthreads();

// in-place reduction in global memory
if (blockDim.x >= 1024 && tid < 512)  smem[tid] += smem[tid + 512];

__syncthreads();

if (blockDim.x >= 512 && tid < 256)  smem[tid] += smem[tid + 256];

__syncthreads();

if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];

__syncthreads();

if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];

__syncthreads();

// unrolling warp
if (tid < 32)
{
volatile int *vsmem = smem;
vsmem[tid] += vsmem[tid + 32];
vsmem[tid] += vsmem[tid + 16];
vsmem[tid] += vsmem[tid +  8];
vsmem[tid] += vsmem[tid +  4];
vsmem[tid] += vsmem[tid +  2];
vsmem[tid] += vsmem[tid +  1];
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = smem[0];
}