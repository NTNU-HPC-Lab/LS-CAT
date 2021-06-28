#include "includes.h"
__global__ void kSoftMaxCrossEntropy(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float tiny) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < width; i += numThreads) {
target[i] = -__logf(mat[height * i + (int)labels[i]] + tiny);
}
}