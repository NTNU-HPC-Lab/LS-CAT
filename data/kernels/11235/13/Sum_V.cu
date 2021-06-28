#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Sum_V(const float* a, float* partial_sums, const int n)
{
extern __shared__ float sdata[];

int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x;
int ti = threadIdx.x;

// move global input data to shared memory, pad with zeros
float x = 0.0f;
if (i < n)
{
x = a[i];
}
sdata[ti] = x;

__syncthreads();

// use parallel reduction to contiguously reduce to partial sums
for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
{
if (ti < offset)
{
sdata[ti] += sdata[ti + offset];
}

__syncthreads();
}

if (ti == 0)
{
partial_sums[blockIdx.x] = sdata[0];
}
}