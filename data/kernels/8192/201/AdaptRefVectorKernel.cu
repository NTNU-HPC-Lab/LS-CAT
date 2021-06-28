#include "includes.h"
__global__ void AdaptRefVectorKernel(  int cell, float *referenceVector, float oldErrorFraction, float youngErrorFraction, float decayFactor, int *winningCount, float *difference, int inputSize  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < inputSize)
{
float errorFraction = (youngErrorFraction - oldErrorFraction) * expf( - decayFactor * winningCount[cell] ) + oldErrorFraction;
referenceVector[cell * inputSize + threadId] += errorFraction * difference[cell * inputSize + threadId];
}
}