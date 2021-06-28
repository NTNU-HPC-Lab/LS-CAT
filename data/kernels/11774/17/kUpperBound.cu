#include "includes.h"
__global__ void kUpperBound(float* mat1, float* mat2, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < len; i += numThreads) {
target[i] = mat1[i] > mat2[i] ? mat2[i] : mat1[i];
}
}