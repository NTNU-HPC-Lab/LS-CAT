#include "includes.h"
__global__ void AddLocalErrorKernel(  int s1, float *distance, float *localError  )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
localError[s1] += distance[s1] * distance[s1];
}

}