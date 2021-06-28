#include "includes.h"
__global__ void kAdagrad(float *history, float *grad, float delta, int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < len; i += numThreads) {
float curr_norm = history[i] - delta;
history[i] = delta + sqrt(curr_norm * curr_norm + grad[i] * grad[i]);
}
}