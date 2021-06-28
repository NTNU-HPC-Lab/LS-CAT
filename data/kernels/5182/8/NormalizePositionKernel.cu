#include "includes.h"
__global__ void NormalizePositionKernel( float *input, float *normalized, float xMax, float yMax )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < 1)
{
normalized[0] = input[0] / xMax;
normalized[1] = input[1] / yMax;
}
}