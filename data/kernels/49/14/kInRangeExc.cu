#include "includes.h"
__global__ void kInRangeExc(float* gData, float lower, float upper, float* target, unsigned int numElements) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
target[i] = gData[i] > lower && gData[i] < upper;
}