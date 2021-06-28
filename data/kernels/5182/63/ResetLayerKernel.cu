#include "includes.h"
__global__ void ResetLayerKernel( float *layer, float value, int count )
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if(threadId < count)
{
layer[threadId] = value;
}

}