#include "includes.h"
__global__ void FullyConnectedUpdateMemoryKernel( float *avgWeightGradPtr, float *avgBiasGradPtr, float *avgWeightGradVarPtr, float *avgBiasGradVarPtr, float *weightMemorySizePtr, float *biasMemorySizePtr, float *dropoutMaskPtr, int prevLayerSize, int thisLayerSize )
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
int index = j;
for (i = 0; i < prevLayerSize; i++)
{
// update memory size
weightMemorySizePtr[index] = (1.0f - avgWeightGradPtr[index] * avgWeightGradPtr[index] / avgWeightGradVarPtr[index]) * weightMemorySizePtr[index] + 1.0f;

index += thisLayerSize;
}

// update memory size
biasMemorySizePtr[j] = (1.0f - avgBiasGradPtr[j] * avgBiasGradPtr[j] / avgBiasGradVarPtr[j]) * biasMemorySizePtr[j] + 1.0f;
}
}
}