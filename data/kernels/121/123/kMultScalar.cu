#include "includes.h"
__global__ void kMultScalar(float* mat, float alpha, float* dest, unsigned int len, float scale_targets) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
if (scale_targets == 0) {
for (unsigned int i = idx; i < len; i += numThreads) {
dest[i] = alpha * mat[i];
}
} else {
for (unsigned int i = idx; i < len; i += numThreads) {
dest[i] = scale_targets * dest[i] + alpha * mat[i];
}
}
}