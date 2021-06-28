#include "includes.h"
__global__ void kLessThanEqScalar(float* mat, float val, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads) target[i] = mat[i] <= val;
}