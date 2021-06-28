#include "includes.h"
__global__ void InterpolateFromMemBlock(float* input1, float* input2, float* output, float* weightMemBlock, int inputSize)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < inputSize)
{
if (weightMemBlock[0] <= 0)
{
output[threadId] = input1[threadId];
}
else if (weightMemBlock[0] >= 1)
{
output[threadId] = input2[threadId];
}
else
{
output[threadId] = (1 - weightMemBlock[0]) * input1[threadId] + weightMemBlock[0] * input2[threadId];
}
}
}