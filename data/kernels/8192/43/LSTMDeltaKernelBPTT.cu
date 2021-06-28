#include "includes.h"
__global__ void LSTMDeltaKernelBPTT( float* deltas, float* cellStates, float* previousCellStates, float* cellStateErrors, float* nextCellStateErrors,  float* outputGateDeltas, float* forgetGateDeltas, float* nextForgetGateDeltas, float* inputGateDeltas, float* nextInputGateDeltas, float* cellInputDeltas,  float* cellInputActivations, float* cellStateActivations, float* outputGateActivations, float* nextForgetGateActivations, float* inputGateActivations,  float* cellInputActivationDerivatives, float* cellStateActivationDerivatives, float* outputGateActivationDerivatives, float* forgetGateActivationDerivatives, float* inputGateActivationDerivatives,  float* cellInputWeights, float* outputGateWeights, float* forgetGateWeights, float* inputGateWeights,  int inputCount, int cellCount, int cellsPerBlock )
{
int memoryBlockId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (memoryBlockId < cellCount / cellsPerBlock)
{
outputGateDeltas[memoryBlockId] = 0;
for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
{
outputGateDeltas[memoryBlockId] += cellStateActivations[cellId] * deltas[cellId];
}
outputGateDeltas[memoryBlockId] *= outputGateActivationDerivatives[memoryBlockId];

for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
{
int relativeCellId = cellId - (memoryBlockId * cellsPerBlock);
int peepHoleWeightId = (memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1)) + inputCount + cellCount + relativeCellId;
cellStateErrors[cellId] = deltas[cellId] * outputGateActivations[memoryBlockId] * cellStateActivationDerivatives[cellId] +
nextCellStateErrors[cellId] * nextForgetGateActivations[memoryBlockId] +
nextInputGateDeltas[memoryBlockId] * inputGateWeights[peepHoleWeightId] +
nextForgetGateDeltas[memoryBlockId] * forgetGateWeights[peepHoleWeightId] +
outputGateDeltas[memoryBlockId] * outputGateWeights[peepHoleWeightId];

cellInputDeltas[cellId] = inputGateActivations[memoryBlockId] * cellInputActivationDerivatives[cellId] * cellStateErrors[cellId];
}

inputGateDeltas[memoryBlockId] = 0;
forgetGateDeltas[memoryBlockId] = 0;
for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
{
inputGateDeltas[memoryBlockId] += cellStateErrors[cellId] * cellInputActivations[cellId];
forgetGateDeltas[memoryBlockId] += cellStateErrors[cellId] * previousCellStates[cellId];
}
inputGateDeltas[memoryBlockId] *= inputGateActivationDerivatives[memoryBlockId];
forgetGateDeltas[memoryBlockId] *= forgetGateActivationDerivatives[memoryBlockId];
}
}