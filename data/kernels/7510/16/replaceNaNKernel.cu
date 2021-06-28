#include "includes.h"
__global__ void replaceNaNKernel( int numberEntriesPerInstance, int numberIterations, float* source, float* destination) {

int instanceIndex = blockIdx.x;

int instanceStart = instanceIndex * numberEntriesPerInstance;
int startWithinThread = instanceStart + blockIdx.y * blockDim.x * numberIterations + threadIdx.x * numberIterations;

for(int index = startWithinThread; index < min(startWithinThread + numberIterations, instanceStart + numberEntriesPerInstance); index++) {
float currentValue = source[index];
destination[index] = isnan(currentValue) ? 0.0f : currentValue;
}
}