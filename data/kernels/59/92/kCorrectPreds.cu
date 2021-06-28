#include "includes.h"
__global__ void kCorrectPreds(float* mat, float* p, float* target, unsigned int len, float cutoff) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads)
target[i] = mat[i] * (p[i] >= cutoff) + (1 - mat[i]) * (p[i] < cutoff);
}