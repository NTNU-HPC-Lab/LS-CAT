#include "includes.h"
__global__ static void timedReduction(const float *input, float *output, clock_t *timer)
{
// __shared__ float shared[2 * blockDim.x];
extern __shared__ float shared[];

const int tid = threadIdx.x;
const int bid = blockIdx.x;

if (tid == 0) timer[bid] = clock();

// Copy input.
shared[tid] = input[tid];
shared[tid + blockDim.x] = input[tid + blockDim.x];

// Perform reduction to find minimum.
for (int d = blockDim.x; d > 0; d /= 2)
{
__syncthreads();

if (tid < d)
{
float f0 = shared[tid];
float f1 = shared[tid + d];

if (f1 < f0)
{
shared[tid] = f1;
}
}
}

// Write result.
if (tid == 0) output[bid] = shared[0];

__syncthreads();

if (tid == 0) timer[bid+gridDim.x] = clock();
}