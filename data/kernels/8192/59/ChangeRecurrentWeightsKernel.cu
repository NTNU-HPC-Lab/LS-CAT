#include "includes.h"
__global__ void ChangeRecurrentWeightsKernel( float *recurrentWeights, float *recurrentWeightDeltas, float *outputWeights, float *outputDeltas, float *recurrentWeightRTRLDerivatives,  float trainingRate, float momentum )
{
int weightId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (weightId < D_HIDDEN_UNITS * D_HIDDEN_UNITS)
{
float gradient = 0;

for (int i = 0; i < D_OUTPUT_UNITS; i++)
{
float sum = 0;
for (int j = 0; j < D_HIDDEN_UNITS; j++)
{
sum += outputWeights[i * D_HIDDEN_UNITS + j] * recurrentWeightRTRLDerivatives[j * D_HIDDEN_UNITS * D_HIDDEN_UNITS + weightId];
}

gradient += outputDeltas[i] * sum;
}

float weightDelta = trainingRate * gradient + momentum * recurrentWeightDeltas[weightId];
recurrentWeightDeltas[weightId] = weightDelta;
recurrentWeights[weightId] += weightDelta;
}
}