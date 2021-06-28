#include "includes.h"
__global__ void InvolveVector(float* input, float* output, int inputSize)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < inputSize - 1)
{
output[0] = input[0];
output[threadId + 1] = input[inputSize - threadId - 1];
}
}