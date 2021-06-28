#include "includes.h"
__global__ void normalise(float* result, unsigned int resultLength, float* amps, unsigned int* hits)
{
int absoluteThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

if(absoluteThreadIdx > resultLength)
return;

result[absoluteThreadIdx] = amps[absoluteThreadIdx] / hits[absoluteThreadIdx / 4];
}