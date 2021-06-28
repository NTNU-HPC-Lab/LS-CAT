#include "includes.h"
__global__ void kCopy(float* srcStart, float* destStart, unsigned int copyWidth, unsigned int jumpWidth, unsigned int numElements) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < numElements)
destStart[(idx / copyWidth) * jumpWidth + idx % copyWidth] = srcStart[(idx / copyWidth) * jumpWidth + idx % copyWidth];
}