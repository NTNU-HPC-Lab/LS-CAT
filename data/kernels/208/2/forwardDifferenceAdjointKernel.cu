#include "includes.h"
__global__ void forwardDifferenceAdjointKernel(const int len, const float* source, float* target) {
for (int idx = blockIdx.x * blockDim.x + threadIdx.x + 1; idx < len - 1;
idx += blockDim.x * gridDim.x) {
target[idx] = -source[idx] + source[idx - 1];
}
}