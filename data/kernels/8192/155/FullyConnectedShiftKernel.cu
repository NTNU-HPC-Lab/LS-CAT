#include "includes.h"
__global__ void FullyConnectedShiftKernel( float *weightPtr, float *biasPtr, float *shiftedWeightsPtr, float *shiftedBiasPtr, float *avgWeightGradPtr, float *avgBiasGradPtr, float *dropoutMaskPtr, int prevLayerSize, int thisLayerSize )
{
// i: prev. layer neuron id
// j: current layer neuron id
int i;
int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (j < thisLayerSize)
{
if (!dropoutMaskPtr[j])
{
// weight gradient
int index = j;
for (i = 0; i < prevLayerSize; i++)
{
shiftedWeightsPtr[index] = weightPtr[index] + avgWeightGradPtr[index]; // TODO: Check if it is correct to add here, or if it should be subtracted
index += thisLayerSize;
}

// bias gradient
shiftedBiasPtr[j] = biasPtr[j] - avgBiasGradPtr[j];
}
}
}