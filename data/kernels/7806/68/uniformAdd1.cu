#include "includes.h"
__global__ void uniformAdd1(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex)
{
__shared__ int uni;
if (threadIdx.x == 0)
uni = uniforms[blockIdx.x + blockOffset];

unsigned int address = __mul24(blockIdx.x, blockDim.x) + baseIndex + threadIdx.x;

__syncthreads();

// note one add per thread
g_data[address] += uni;
}