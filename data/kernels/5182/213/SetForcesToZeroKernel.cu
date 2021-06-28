#include "includes.h"
__global__ void SetForcesToZeroKernel(  float *force, int maxCells  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < maxCells * 3)
{
force[threadId] = 0.00f;
}
}