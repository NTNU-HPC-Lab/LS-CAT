#include "includes.h"
__global__ void reduceCompleteUnrollWarps8Sync(int *g_idata, int *g_odata, unsigned int n)
{
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

// convert global data pointer to the local pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x * 8;

// unrolling 8
if (idx + 7 * blockDim.x < n)
{
int a1 = g_idata[idx];
int a2 = g_idata[idx + blockDim.x];
int a3 = g_idata[idx + 2 * blockDim.x];
int a4 = g_idata[idx + 3 * blockDim.x];
int b1 = g_idata[idx + 4 * blockDim.x];
int b2 = g_idata[idx + 5 * blockDim.x];
int b3 = g_idata[idx + 6 * blockDim.x];
int b4 = g_idata[idx + 7 * blockDim.x];
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
idata[tid] += idata[tid + 32];
__syncthreads();

if (tid < 32)
idata[tid] += idata[tid + 16];
__syncthreads();

if (tid < 32)
idata[tid] += idata[tid +  8];
__syncthreads();

if (tid < 32)
idata[tid] += idata[tid +  4];
__syncthreads();

if (tid < 32)
idata[tid] += idata[tid +  2];
__syncthreads();

if (tid < 32)
idata[tid] += idata[tid +  1];

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = idata[0];
}