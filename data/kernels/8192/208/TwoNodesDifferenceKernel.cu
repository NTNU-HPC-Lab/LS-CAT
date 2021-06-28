#include "includes.h"
__global__ void TwoNodesDifferenceKernel(  int nodeOne, int nodeTwo, int vectorLength, float *referenceVector, float *twoNodesDifference  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < vectorLength)
{
twoNodesDifference[threadId] = referenceVector[nodeOne * vectorLength + threadId] - referenceVector[nodeTwo * vectorLength + threadId];
}
}