#include "includes.h"
__global__ void kWhere(float* condition_mat, float* if_mat, float* else_mat, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < len; i += numThreads) {
target[i] = condition_mat[i] ? if_mat[i] : else_mat[i];
}
}