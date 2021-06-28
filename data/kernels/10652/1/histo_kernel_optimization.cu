#include "includes.h"

#define SIZE (100 * 1024 * 1024)



__global__ void histo_kernel_optimization(unsigned char *buffer, int size, unsigned int *histo)
{
__shared__ unsigned int temp[256];
temp[threadIdx.x] = 0;
__syncthreads();

int i = threadIdx.x + blockDim.x * blockIdx.x;
int stride = blockDim.x * gridDim.x;

while (i < size)
{
atomicAdd(&histo[buffer[i]], 1);
i += stride;
}
__syncthreads();
atomicAdd(&histo[threadIdx.x], temp[threadIdx.x]);
}