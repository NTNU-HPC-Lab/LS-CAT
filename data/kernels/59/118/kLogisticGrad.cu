#include "includes.h"
__global__ void kLogisticGrad(float* mat, float* targets, float* out_grad, unsigned int numEls) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < numEls; i += numThreads) {
out_grad[i] = (targets[i] < 0) ? 0 : (mat[i] - targets[i]);
}
}