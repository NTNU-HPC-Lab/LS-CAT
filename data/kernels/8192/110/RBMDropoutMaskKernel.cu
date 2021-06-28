#include "includes.h"
__global__ void RBMDropoutMaskKernel( float *maskPtr, float dropout, int thisLayerSize )
{

int index = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (index < thisLayerSize)
{
maskPtr[index] = dropout < maskPtr[index];
}
}