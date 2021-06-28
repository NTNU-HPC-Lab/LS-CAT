#include "includes.h"

#define N 10000000

__global__ void compute_histogram(unsigned char *data, unsigned int *histogram)
{
__shared__ unsigned int cache[256];
int i = blockIdx.x * blockDim.x + threadIdx.x;

cache[threadIdx.x] = 0;
__syncthreads();

while(i < N)
{
atomicAdd(&cache[data[i]], 1);
i += blockDim.x * gridDim.x;
}

__syncthreads();
atomicAdd(&histogram[threadIdx.x], cache[threadIdx.x]);
}