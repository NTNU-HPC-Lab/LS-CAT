#include "includes.h"
__global__ void DecreaseErrorAndUtilityKernel(  float *localError, float *utility, int *activityFlag, int maxCells, float beta  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells)
{
if(activityFlag[threadId] == 1)
{
localError[threadId] -= beta * localError[threadId];
utility[threadId] -= beta * utility[threadId];
}
}
}