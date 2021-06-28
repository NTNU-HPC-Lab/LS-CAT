#include "includes.h"
__global__ void NegativeCorrelationForwardResetKernel( float* outputPtr, int thisLayerSize )
{
// j: current layer neuron id
int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (j < thisLayerSize)
{
outputPtr[j] = 0;
}
}