#include "includes.h"

#define SIZE (100 * 1024 * 1024)



__global__ void histo_kernel(unsigned char *buffer, int size, unsigned int *histo)
{
int i = threadIdx.x + blockDim.x * blockIdx.x;
int stride = blockDim.x * gridDim.x;

while (i < size)
{
atomicAdd(&histo[buffer[i]], 1);
i += stride;
}
}