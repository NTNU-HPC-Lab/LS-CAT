#include "includes.h"
__global__ void kApplySoftThreshold(float* mat, float alpha, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < len; i += numThreads) {
float f = mat[i];
target[i] = f > 0 ? max(0., f - alpha) : min(0., f + alpha);
}
}