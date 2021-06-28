#include "includes.h"
__global__ void AddUtilityKernel(  int s1, int s2, float *distance, float *utility  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
utility[s1] += distance[s2] - distance[s1];
}
}