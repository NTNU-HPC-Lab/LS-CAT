#include "includes.h"
__global__ void reduceNeighboredSmem(int *g_idata, int *g_odata, unsigned int  n)
{
__shared__ int smem[DIM];

// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

// convert global data pointer to the local pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x;

// boundary check
if (idx >= n) return;

smem[tid] = idata[tid];
__syncthreads();

// in-place reduction in global memory
for (int stride = 1; stride < blockDim.x; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
smem[tid] += smem[tid + stride];
}

// synchronize within threadblock
__syncthreads();
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = smem[0];
}