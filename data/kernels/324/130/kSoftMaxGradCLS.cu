#include "includes.h"
__global__ void kSoftMaxGradCLS(float* mat, int* labels, float* indices, float* target, unsigned int width, unsigned int height) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < width*height; i += numThreads) {
target[i] = mat[i] - (labels[(int)indices[i % height]] == i / height ? 1 : 0);
}
}