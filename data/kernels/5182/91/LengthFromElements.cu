#include "includes.h"
__global__ void LengthFromElements(float* element1, float* element2, float* output, int count)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < count)
{
output[threadId] = sqrtf(element1[threadId] * element1[threadId] + element2[threadId] * element2[threadId]);
}
}