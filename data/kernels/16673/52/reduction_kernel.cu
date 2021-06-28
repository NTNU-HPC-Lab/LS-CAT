#include "includes.h"
__global__ void reduction_kernel(float* d_out, float* d_in, unsigned int size)
{
unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

extern __shared__ float s_data[];

s_data[threadIdx.x] = (idx_x < size) ? d_in[idx_x] : 0.f;

__syncthreads();

// do reduction
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
{
// thread synchronous reduction
if ( (idx_x % (stride * 2)) == 0 )
s_data[threadIdx.x] += s_data[threadIdx.x + stride];

__syncthreads();
}

if (threadIdx.x == 0)
d_out[blockIdx.x] = s_data[0];
}