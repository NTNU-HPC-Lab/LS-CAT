#include "includes.h"
__global__ void LSTMDeltaKernel( float *cellStateErrors, float *outputGateDeltas, float *cellStates, float *outputGateActivations, float *outputGateActivationDerivatives, float *deltas,  int cellCount, int cellsPerBlock )
{
int memoryBlockId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (memoryBlockId < cellCount / cellsPerBlock)
{
float outputGateDeltaSum = 0.0;

for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
{
float delta = deltas[cellId];
cellStateErrors[cellId] = outputGateActivations[memoryBlockId] * delta;
outputGateDeltaSum += cellStates[cellId] * delta;
}

outputGateDeltas[memoryBlockId] = outputGateActivationDerivatives[memoryBlockId] * outputGateDeltaSum;
}
}