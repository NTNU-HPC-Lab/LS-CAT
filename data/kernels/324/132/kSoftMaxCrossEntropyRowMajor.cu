#include "includes.h"
__global__ void kSoftMaxCrossEntropyRowMajor(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float tiny) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < height; i += numThreads) {
target[i] = -__logf(mat[height * (int)labels[i] + i] + tiny);
}
}