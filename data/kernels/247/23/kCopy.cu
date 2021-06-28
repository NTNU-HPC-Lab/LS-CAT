#include "includes.h"
__global__ void kCopy(float* srcStart, float* destStart, const int copyWidth, const int srcJumpWidth, const int destJumpWidth, const int numElements) {
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
for (int i = idx; i < numElements; i += blockDim.x * gridDim.x) {
destStart[(i / copyWidth) * destJumpWidth + i % copyWidth] = srcStart[(i / copyWidth) * srcJumpWidth + i % copyWidth];
}
}