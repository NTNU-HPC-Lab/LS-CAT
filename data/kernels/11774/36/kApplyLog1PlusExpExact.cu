#include "includes.h"
__global__ void kApplyLog1PlusExpExact(float* mat, float* target, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
float mat_i;

for (unsigned int i = idx; i < len; i += numThreads) {
mat_i = mat[i];
if (mat_i > 0)
target[i] = (log(1 + exp(-mat_i)) + mat_i);
else
target[i] = log(1 + exp(mat_i));
}
}