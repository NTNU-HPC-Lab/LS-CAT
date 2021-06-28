#include "includes.h"
__global__ void reduction_kernel(float *g_out, float *g_in, unsigned int size)
{
unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

extern __shared__ float s_data[];

// cumulates input with grid-stride loop and save to share memory
float input[NUM_LOAD] = {0.f};
for (int i = idx_x; i < size; i += blockDim.x * gridDim.x * NUM_LOAD)
{
for (int step = 0; step < NUM_LOAD; step++)
input[step] += (i + step * blockDim.x * gridDim.x < size) ? g_in[i + step * blockDim.x * gridDim.x] : 0.f;
}
for (int i = 1; i < NUM_LOAD; i++)
input[0] += input[i];
s_data[threadIdx.x] = input[0];

__syncthreads();

// do reduction
for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
{
if (threadIdx.x < stride)
s_data[threadIdx.x] += s_data[threadIdx.x + stride];

__syncthreads();
}

if (threadIdx.x == 0) {
g_out[blockIdx.x] = s_data[0];
}
}