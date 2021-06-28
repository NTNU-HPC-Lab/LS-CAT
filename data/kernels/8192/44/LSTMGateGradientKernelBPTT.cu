#include "includes.h"
__global__ void LSTMGateGradientKernelBPTT( float *input, float *previousOutput, float *cellStates,  float *inputGateDeltas, float *forgetGateDeltas, float *outputGateDeltas,  float* outputGateWeightGradient, float* inputGateWeightGradient, float* forgetGateWeightGradient,  int inputCount, int previousOutputCount, int cellsPerBlock )
{
int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
+ blockDim.x * blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

int weightsPerGate = inputCount + previousOutputCount + cellsPerBlock + 1;

if (weightId < weightsPerGate * previousOutputCount / cellsPerBlock)
{
int fromId = weightId % weightsPerGate;
int toId = weightId / weightsPerGate;

//calculate output gate weight gradient
int isFromInputUnit = fromId >= 0 && fromId < inputCount;
int isFromPreviousOutputUnit = (fromId >= inputCount) && (fromId < inputCount + previousOutputCount);
int isPeephole = (fromId >= inputCount + previousOutputCount) && (fromId < inputCount + previousOutputCount + cellsPerBlock);
int isFromBiasUnit = fromId == (inputCount + previousOutputCount + cellsPerBlock);

float inputFromWeight = isFromInputUnit * input[isFromInputUnit * fromId]
+ isFromPreviousOutputUnit * previousOutput[isFromPreviousOutputUnit * (fromId - inputCount)]
+ isPeephole * cellStates[isPeephole * (toId * cellsPerBlock + (fromId - inputCount - previousOutputCount))]
+ isFromBiasUnit * 1;

outputGateWeightGradient[weightId] = outputGateDeltas[toId] * inputFromWeight;
inputGateWeightGradient[weightId] = inputGateDeltas[toId] * inputFromWeight;
forgetGateWeightGradient[weightId] = forgetGateDeltas[toId] * inputFromWeight;
}
}