#include "includes.h"
__global__ void child_kernel(int *data, int seed)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

atomicAdd(&data[idx], seed);
}