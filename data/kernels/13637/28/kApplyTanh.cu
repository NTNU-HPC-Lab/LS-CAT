#include "includes.h"
__global__ void kApplyTanh(float* mat, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
float mat_i, exp2x;

for (unsigned int i = idx; i < len; i += numThreads) {
mat_i = mat[i];
exp2x = __expf(2 * mat_i);
target[i] = 1 - 2 / (exp2x + 1);
}
}