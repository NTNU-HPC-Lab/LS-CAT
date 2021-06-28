#include "includes.h"
__global__ void reduceUnrolling (int *g_idata, int *g_odata, unsigned int n, unsigned int q) //added int q
{
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x * q + threadIdx.x; // q adapted idx

// unroll analogous q
if (idx + blockDim.x*(q-1) < n)
{
for (int i=1; i<q; i++)
{
g_idata[idx] += g_idata[idx + blockDim.x*i];
}
}
__syncthreads();

// in-place reduction in global memory
for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
{
if (tid < stride)
{
g_idata[idx] += g_idata[idx + stride];
}

// synchronize within threadblock
__syncthreads();
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = g_idata[idx];
}