#include "includes.h"
__global__ void reduceCompUnrollW(int *g_idata, int *g_odata, unsigned int n)
{
// set the thread id.
unsigned int tid = threadIdx.x;
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;

// convert global data pointer to the local pointer of this block.
int *idata = g_idata + blockIdx.x * blockDim.x * 8;

// unrolling 8 data blocks.
if (idx + blockDim.x * 7 < n)
{
int a1 = g_idata[idx];
int a2 = g_idata[idx + blockDim.x];
int a3 = g_idata[idx + blockDim.x * 2];
int a4 = g_idata[idx + blockDim.x * 3];
int b1 = g_idata[idx + blockDim.x * 4];
int b2 = g_idata[idx + blockDim.x * 5];
int b3 = g_idata[idx + blockDim.x * 6];
int b4 = g_idata[idx + blockDim.x * 7];
g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
}
__syncthreads();

// in-place reduction and complete unroll
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
volatile int *vmem = idata;
vmem[tid] += vmem[tid + 32];
vmem[tid] += vmem[tid + 16];
vmem[tid] += vmem[tid +  8];
vmem[tid] += vmem[tid +  4];
vmem[tid] += vmem[tid +  2];
vmem[tid] += vmem[tid +  1];
}

// write result for this block to global mem.
if (tid == 0)
{
g_odata[blockIdx.x] = idata[0];
}
}