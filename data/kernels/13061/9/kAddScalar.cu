#include "includes.h"
__global__ void kAddScalar(float* gData, float scalar, float* target, unsigned int numElements) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

for (unsigned int i = idx; i < numElements; i += blockDim.x * gridDim.x)
target[i] = scalar + gData[i];
}