#include "includes.h"
__global__ void gpuDot(float* dot, float* a, float* b, int N)
{
__shared__ float cache[THREADS_PER_BLOCK];
int tid = blockIdx.x*blockDim.x + threadIdx.x;
int cacheIdx = threadIdx.x;

float temp = 0;

while (tid < N)
{
temp += a[tid] * b[tid];
tid += blockDim.x * gridDim.x;
}

cache[cacheIdx]=temp;

__syncthreads();

int i = blockDim.x/2;
while (i != 0)
{
if (cacheIdx < i)
cache[cacheIdx] += cache[cacheIdx + i];

__syncthreads();
i /= 2;
}

if (cacheIdx == 0)
dot[blockIdx.x] = cache[0];
}