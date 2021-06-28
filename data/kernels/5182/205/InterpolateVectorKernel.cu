#include "includes.h"
__global__ void InterpolateVectorKernel(  int r, int q, int f, int inputSize, float *referenceVector  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < inputSize)
{
referenceVector[r * inputSize + threadId] = 0.50f * (referenceVector[q * inputSize + threadId] + referenceVector[f * inputSize + threadId]);
}
}