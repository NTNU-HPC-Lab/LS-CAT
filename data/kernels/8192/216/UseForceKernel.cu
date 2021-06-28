#include "includes.h"
__global__ void UseForceKernel(  float *force, float forceFactor, float *pointsCoordinates, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells * 3)
{
pointsCoordinates[threadId] += forceFactor * force[threadId];
}
}