#include "includes.h"
__global__ void Dot(float *a, float *b, float *c)
{
__shared__ float cache[ThreadsPerBlock];
int tid = threadIdx.x + blockIdx.x * blockDim.x;
int cacheIndex = threadIdx.x;
float temp = 0;
while (tid < N) {
temp += a[tid] * b[tid];
tid += blockDim.x * gridDim.x;
}
cache[cacheIndex] = temp;
__syncthreads();

int i = blockDim.x / 2;
while (i != 0)
{
if (cacheIndex < i)
cache[cacheIndex] += cache[cacheIndex + i];
__syncthreads();
i /= 2;
}

if (cacheIndex == 0)
c[blockIdx.x] = cache[0];
}