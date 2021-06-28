#include "includes.h"
__global__ void VectorInputDiffKernel(  float *input, int inputSize, float *referenceVector, int maxCells, float *difference  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells * inputSize)
{
difference[threadId] = input[threadId % inputSize] - referenceVector[threadId];
}
}