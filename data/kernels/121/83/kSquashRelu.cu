#include "includes.h"
__global__ void kSquashRelu(float* mat, float* target, unsigned int len, float lambda) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads) target[i] = 2 / (1 + __expf(-lambda * mat[i])) - 1;
}