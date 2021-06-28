#include "includes.h"
__global__ void kCrossEntropyBernoulli(float* mat, float* p, float* target, unsigned int len, float tiny) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads)
target[i] = -mat[i] * __logf(p[i] + tiny) - (1 - mat[i]) * __logf(1 - p[i] + tiny);
}