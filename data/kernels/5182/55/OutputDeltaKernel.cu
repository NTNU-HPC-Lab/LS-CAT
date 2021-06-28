#include "includes.h"
__global__ void OutputDeltaKernel(float *outputDeltas, float *target, float *outputActivations, float *outputActivationDerivatives)
{
int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;
if (unitId < D_OUTPUT_UNITS)
{
outputDeltas[unitId] = (target[unitId] - outputActivations[unitId]) * outputActivationDerivatives[unitId];
}
}