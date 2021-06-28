#include "includes.h"
__global__ void dot(float *a, float *b, float *c)
{
__shared__ float cache[threadsPerBlock];
int cacheIndex = threadIdx.x;

float temp = 0.0;
for (int tid = threadIdx.x + blockIdx.x*blockDim.x; tid<N; tid += blockDim.x*gridDim.x)
{
temp += a[tid]*b[tid];
}

cache[cacheIndex] = temp;

__syncthreads();


// reduction
for (int i = blockDim.x/2; i>0; i /= 2)
{
if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
__syncthreads();
}

if (threadIdx.x == 0) c[blockIdx.x] = cache[0];
}