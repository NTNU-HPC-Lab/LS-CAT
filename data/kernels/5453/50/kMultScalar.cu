#include "includes.h"
__global__ void kMultScalar(float* mat, float alpha, float* dest, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < len; i += numThreads) {
dest[i] = alpha * mat[i];
}
}