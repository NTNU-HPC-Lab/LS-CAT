#include "includes.h"
__global__ void childKernel(unsigned int parentThreadIndex, float* data)
{
printf("Parent thread index: %d, child thread index: %d\n",
parentThreadIndex, threadIdx.x);
data[threadIdx.x] = parentThreadIndex + 0.1f * threadIdx.x;
}