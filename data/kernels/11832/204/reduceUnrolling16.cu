#include "includes.h"
__global__ void reduceUnrolling16 (int *g_idata, int *g_odata, unsigned int n)
{
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

// convert global data pointer to the local pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x * 16;

// unrolling 16
if (idx + 15 * blockDim.x < n)
{
int a1 = g_idata[idx];
int a2 = g_idata[idx + blockDim.x];
int a3 = g_idata[idx + 2 * blockDim.x];
int a4 = g_idata[idx + 3 * blockDim.x];
int b1 = g_idata[idx + 4 * blockDim.x];
int b2 = g_idata[idx + 5 * blockDim.x];
int b3 = g_idata[idx + 6 * blockDim.x];
int b4 = g_idata[idx + 7 * blockDim.x];
int c1 = g_idata[idx + 8 * blockDim.x];
int c2 = g_idata[idx + 9 * blockDim.x];
int c3 = g_idata[idx + 10 * blockDim.x];
int c4 = g_idata[idx + 11 * blockDim.x];
int d1 = g_idata[idx + 12 * blockDim.x];
int d2 = g_idata[idx + 13 * blockDim.x];
int d3 = g_idata[idx + 14 * blockDim.x];
int d4 = g_idata[idx + 15 * blockDim.x];
g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 + c1 + c2 + c3 + c4
+ d1 + d2 + d3 + d4;
}

__syncthreads();

// in-place reduction in global memory
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
{
if (tid < stride)
{
idata[tid] += idata[tid + stride];
}

// synchronize within threadblock
__syncthreads();
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = idata[0];
}