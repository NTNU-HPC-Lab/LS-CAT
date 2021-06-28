#include "includes.h"
__global__ void copyKernel(float* from, float* to, int size)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

if(threadId < size)
{
to[threadId] = from[threadId];
}
}