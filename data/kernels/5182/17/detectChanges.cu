#include "includes.h"
__global__ void detectChanges(float* a, float* b, float* result, int size, float value)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x
+ blockDim.x*blockIdx.x
+ threadIdx.x;

if(threadId < size)
{
if(a[threadId] > b[threadId])
{
result[threadId] = value;
}
else if(a[threadId] <b[threadId])
{
result[threadId] = -value;
}
else
{
result[threadId] = 0;
}
}
}