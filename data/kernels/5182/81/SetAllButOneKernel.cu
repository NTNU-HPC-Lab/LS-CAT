#include "includes.h"
__global__ void SetAllButOneKernel(float *buffer, float value, int index, int count)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < count && threadId != index)
{
buffer[threadId] = value;
}

}