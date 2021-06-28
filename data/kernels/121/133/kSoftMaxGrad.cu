#include "includes.h"
__global__ void kSoftMaxGrad(float* mat, float* labels, float* target, unsigned int width, unsigned int height) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < width * height; i += numThreads) {
target[i] = mat[i] - ((int)labels[i / height] == i % height ? 1 : 0);
}
}