#include "includes.h"
__global__ void RecurrentWeightsRTRLDerivativesKernel( float *previousHiddenActivations, float *hiddenActivationDerivatives, float *recurrentWeights, float *recurrentWeightRTRLDerivatives, float *previousRecurrentWeightRTRLDerivatives )
{
int partialId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (partialId < D_HIDDEN_UNITS * D_HIDDEN_UNITS * D_HIDDEN_UNITS)
{
int unitId = partialId / (D_HIDDEN_UNITS * D_HIDDEN_UNITS);
int weightId = partialId % (D_HIDDEN_UNITS * D_HIDDEN_UNITS);
int to = weightId / D_HIDDEN_UNITS;
int from = weightId % D_HIDDEN_UNITS;

float sum = 0;
for (int i = 0; i < D_HIDDEN_UNITS; i++)
{
sum += recurrentWeights[unitId * D_HIDDEN_UNITS + i] * previousRecurrentWeightRTRLDerivatives[i * (D_HIDDEN_UNITS * D_HIDDEN_UNITS) + weightId];
}

recurrentWeightRTRLDerivatives[partialId] = hiddenActivationDerivatives[unitId] * ((unitId == to) * previousHiddenActivations[from] + sum);
}
}