#include "includes.h"
__global__ void vectorAddUniform4(uint *d_vector, const uint *d_uniforms, const int n)
{
__shared__ uint uni[1];

if (threadIdx.x == 0)
{
uni[0] = d_uniforms[blockIdx.x];
}

unsigned int address = threadIdx.x + (blockIdx.x *
blockDim.x * 4);

__syncthreads();

// 4 elems per thread
for (int i = 0; i < 4 && address < n; i++)
{
d_vector[address] += uni[0];
address += blockDim.x;
}
}