#include "includes.h"
__global__ void kDivideScalar(float* mat, float alpha, float* dest, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < len; i += numThreads) {
dest[i] = mat[i] / alpha;
}
}