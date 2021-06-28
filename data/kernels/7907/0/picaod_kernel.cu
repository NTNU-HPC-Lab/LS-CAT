#include "includes.h"
__global__ void picaod_kernel(unsigned int *dev_v, long size, unsigned int *temp)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

int offset = x + y * blockDim.x * gridDim.x;

atomicAdd(&(temp[0]), dev_v[offset]);

}