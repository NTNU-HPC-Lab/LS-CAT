#include "includes.h"
__global__ void kRMSProp(float *history, float *grad, float factor, int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads) {
history[i] = sqrt(factor * history[i] * history[i] + (1-factor) * grad[i] * grad[i]);
}
}