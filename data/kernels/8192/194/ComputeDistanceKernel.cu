#include "includes.h"
__global__ void ComputeDistanceKernel(  int inputSize, float *distance, float *dimensionWeight, int maxCells, float *difference  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
float sum = 0.00f;
float value;
for(int i = 0; i < inputSize; i++)
{
value = difference[threadId * inputSize + i];
sum += dimensionWeight[i] * value*value;
}
distance[threadId] = sqrtf(sum);
}
}