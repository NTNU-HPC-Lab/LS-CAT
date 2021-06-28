#include "includes.h"
__global__ void reduction_kernel_1(float* g_out, float* g_in, unsigned int size)
{
unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

extern __shared__ float s_data[];

s_data[threadIdx.x] = (idx_x < size) ? g_in[idx_x] : 0.f;

__syncthreads();

// do reduction
// interleaved addressing
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
{
int index = 2 * stride * threadIdx.x;

if (index < blockDim.x)
s_data[index] += s_data[index + stride];

__syncthreads();
}

if (threadIdx.x == 0)
g_out[blockIdx.x] = s_data[0];
}