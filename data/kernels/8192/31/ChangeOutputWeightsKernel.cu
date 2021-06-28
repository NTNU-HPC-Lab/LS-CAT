#include "includes.h"
__global__ void ChangeOutputWeightsKernel( float *outputWeights, float *outputWeightDeltas, float *outputDeltas, float *hiddenActivations,  float trainingRate, float momentum )
{
int weightId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

int to = weightId / D_HIDDEN_UNITS;
int from = weightId % D_HIDDEN_UNITS;

if (weightId < D_OUTPUT_UNITS * D_HIDDEN_UNITS)
{
float gradient = outputDeltas[to] * hiddenActivations[from];
float weightDelta = trainingRate * gradient + momentum * outputWeightDeltas[weightId];
outputWeightDeltas[weightId] = weightDelta;
outputWeights[weightId] += weightDelta;
}
}