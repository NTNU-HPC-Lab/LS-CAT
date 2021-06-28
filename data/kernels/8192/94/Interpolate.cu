#include "includes.h"
__global__ void Interpolate(float* input1, float* input2, float* output, float weight, int inputSize)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < inputSize)
{
if (weight <= 0)
{
output[threadId] = input1[threadId];
}
else if (weight >= 1)
{
output[threadId] = input2[threadId];
}
else
{
output[threadId] = (1 - weight) * input1[threadId] + weight * input2[threadId];
}
}
}