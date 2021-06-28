#include "includes.h"
__global__ void FullyConnectedAdjustMemoryKernel( float *weightsGradPtr, float *biasGradPtr, float *weightGradCurvePtr, float *biasGradCurvePtr, float *avgWeightGradPtr, float *avgBiasGradPtr, float *avgWeightGradVarPtr, float *avgBiasGradVarPtr, float *avgWeightGradCurvePtr, float *avgBiasGradCurvePtr, float *avgWeightGradCurveVarPtr, float *avgBiasGradCurveVarPtr, float *weightMemorySizePtr, float *biasMemorySizePtr, float *dropoutMaskPtr, int prevLayerSize, int thisLayerSize )
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
// check for weight outliers
if (
abs(weightsGradPtr[index] - avgWeightGradPtr[index]) > 2 * sqrtf(avgWeightGradVarPtr[index]) - avgWeightGradPtr[index] * avgWeightGradPtr[index] ||
abs(weightGradCurvePtr[index] - avgWeightGradCurvePtr[index]) > 2 * sqrtf(avgWeightGradCurveVarPtr[index] - avgWeightGradCurveVarPtr[index] * avgWeightGradCurveVarPtr[index])
)
// TODO: test which one works best
//weightMemorySizePtr[index] += 1; // original method suggested in http://arxiv.org/pdf/1301.3764.pdf
weightMemorySizePtr[index] = 2.2f; // reset to 2.2 according to the Adasecant method in http://arxiv.org/pdf/1412.7419v4.pdf

index += thisLayerSize;
}

// check for bias outliers
if (
abs(biasGradPtr[j] - avgBiasGradPtr[j]) > 2 * sqrtf(avgBiasGradVarPtr[j]) - avgBiasGradPtr[j] * avgBiasGradPtr[j] ||
abs(biasGradCurvePtr[j] - avgBiasGradCurvePtr[j]) > 2 * sqrtf(avgBiasGradCurveVarPtr[j] - avgBiasGradCurveVarPtr[j] * avgBiasGradCurveVarPtr[j])
)
// TODO: test which one works best
//biasMemorySizePtr[j] += 1; // original method suggested in http://arxiv.org/pdf/1301.3764.pdf
biasMemorySizePtr[j] = 2.2f; // reset to 2.2 according to the Adasecant method in http://arxiv.org/pdf/1412.7419v4.pdf
}
}
}