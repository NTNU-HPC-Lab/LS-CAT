#include "includes.h"
__global__ void RBMInputForwardKernel( float *inputPtr, float *outputPtr, float *biasPtr, bool applyBias, int thisLayerSize )
{
// i: current neuron id
int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (i < thisLayerSize)
{
float result = inputPtr[i];
if (applyBias)
result += biasPtr[i];
outputPtr[i] = inputPtr[i];
}
}