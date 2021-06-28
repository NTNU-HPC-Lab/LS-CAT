#include "includes.h"
__device__ inline float sigmoid(float x) {
return 1.0f / (1.0f + __expf(-x));
}
__global__ void kApplySigmoid(float* mat, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads) target[i] = sigmoid(mat[i]);
}