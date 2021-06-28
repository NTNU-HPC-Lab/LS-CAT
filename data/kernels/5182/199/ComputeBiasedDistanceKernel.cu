#include "includes.h"
__global__ void ComputeBiasedDistanceKernel(  float *distance, float *biasedDistance, float *biasTerm, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
biasedDistance[threadId] = distance[threadId] + biasTerm[threadId];
}
}