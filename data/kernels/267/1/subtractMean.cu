#include "includes.h"
__global__ void subtractMean(float* d_inputArray, uint64_t inputLength, float d_mean)
{
uint32_t globalThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;

if(globalThreadIndex >= inputLength)
return;

d_inputArray[globalThreadIndex] -= d_mean;
}