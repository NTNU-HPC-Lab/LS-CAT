#include "includes.h"
__global__ void InputWeightsRTRLDerivativesKernel( float *input, float *hiddenActivationDerivatives, float *recurrentWeights, float *inputWeightRTRLDerivatives, float *previousInputWeightRTRLDerivatives )
{
int partialId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (partialId < D_HIDDEN_UNITS * D_HIDDEN_UNITS * D_INPUT_UNITS)
{
int unitId = partialId / (D_HIDDEN_UNITS * D_INPUT_UNITS);
int weightId = partialId % (D_HIDDEN_UNITS * D_INPUT_UNITS);
int to = weightId / D_INPUT_UNITS;
int from = weightId % D_INPUT_UNITS;

float sum = 0;
for (int i = 0; i < D_HIDDEN_UNITS; i++)
{
sum += recurrentWeights[unitId * D_HIDDEN_UNITS + i] * previousInputWeightRTRLDerivatives[i * (D_HIDDEN_UNITS * D_INPUT_UNITS) + weightId];
}

inputWeightRTRLDerivatives[partialId] = hiddenActivationDerivatives[unitId] * ((unitId == to) * input[from] + sum);
}
}