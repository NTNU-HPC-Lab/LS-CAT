#include "includes.h"
__global__ void kLogistic2(float* gData, float* target, unsigned int numElements) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
target[i] = 1 / (1 + expf(-gData[i]));
}