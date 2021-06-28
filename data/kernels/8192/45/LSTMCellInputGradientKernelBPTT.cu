#include "includes.h"
__global__ void LSTMCellInputGradientKernelBPTT( float *input, float *previousOutput,  float *cellInputDeltas, float *cellInputWeightGradient,  int inputCount, int previousOutputCount, int cellsPerBlock )
{
int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

int weightsPerCell = inputCount + previousOutputCount + 1;

if (weightId < weightsPerCell * previousOutputCount)
{
int fromId = weightId % weightsPerCell;
int toId = weightId / weightsPerCell;

int isFromInputUnit = fromId >= 0 && fromId < inputCount;
int isFromPreviousOutputUnit = (fromId >= inputCount) && (fromId < inputCount + previousOutputCount);
int isFromBiasUnit = fromId == (inputCount + previousOutputCount);

float inputFromWeight = isFromInputUnit * input[isFromInputUnit * fromId]
+ isFromPreviousOutputUnit * previousOutput[isFromPreviousOutputUnit * (fromId - inputCount)]
+ isFromBiasUnit * 1;

cellInputWeightGradient[weightId] = cellInputDeltas[toId] * inputFromWeight;
}
}