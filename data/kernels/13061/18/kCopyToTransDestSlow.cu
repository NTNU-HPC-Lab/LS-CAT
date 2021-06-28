#include "includes.h"
__device__ inline int getTransArrayIndex(unsigned int width, unsigned int height, unsigned  int i) {
return height * (i % width) + i / width;
}
__global__ void kCopyToTransDestSlow(float* srcStart, float* destStart, unsigned int srcCopyWidth, unsigned int srcJumpWidth, unsigned int destJumpHeight, unsigned int numElements) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx < numElements)
destStart[getTransArrayIndex(srcCopyWidth, destJumpHeight, idx)] = srcStart[(idx / srcCopyWidth) * srcJumpWidth + idx % srcCopyWidth];
}