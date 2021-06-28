#include "includes.h"
__global__ void compute_histogram(unsigned char *data, unsigned int *histogram)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

while(i < N)
{
atomicAdd(&histogram[data[i]], 1);
i += blockDim.x * gridDim.x;
}
}