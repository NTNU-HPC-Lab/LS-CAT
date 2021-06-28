#include "includes.h"
__global__ void stencil_1d(float *in, float *out, int N)
{
// shared memory
__shared__ float smem[BDIM + 2 * RADIUS];

// index to global memory
int idx = blockIdx.x * blockDim.x + threadIdx.x;

while (idx < N)
{

// index to shared memory for stencil calculatioin
int sidx = threadIdx.x + RADIUS;

// Read data from global memory into shared memory
smem[sidx] = in[idx];

// read halo part to shared memory
if (threadIdx.x < RADIUS)
{
smem[sidx - RADIUS] = in[idx - RADIUS];
smem[sidx + BDIM] = in[idx + BDIM];
}

// Synchronize (ensure all the data is available)
__syncthreads();

// Apply the stencil
float tmp = 0.0f;

#pragma unroll
for (int i = 1; i <= RADIUS; i++)
{
tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
}

// Store the result
out[idx] = tmp;

idx += gridDim.x * blockDim.x;
}
}