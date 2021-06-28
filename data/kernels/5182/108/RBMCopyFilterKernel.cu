#include "includes.h"
__global__ void RBMCopyFilterKernel( float *weightPtr, float *filterPtr, int weightCount, int i, int thisLayerSize )
{

int weightIndex = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (weightIndex < weightCount)
{
filterPtr[weightIndex] = weightPtr[i + weightIndex * thisLayerSize];
}
}