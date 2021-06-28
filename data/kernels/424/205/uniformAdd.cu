#include "includes.h"
__global__ void uniformAdd(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex)
{
__shared__ int uni;
if (threadIdx.x == 0)
uni = uniforms[blockIdx.x + blockOffset];

unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

__syncthreads();

// note two adds per thread
g_data[address]              += uni;
g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}