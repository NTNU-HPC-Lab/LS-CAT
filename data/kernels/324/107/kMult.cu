#include "includes.h"
__global__ void kMult(float* a, float* b, float* dest, unsigned int numEls, float scale_targets) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
if (scale_targets == 0) {
for (unsigned int i = idx; i < numEls; i += numThreads) {
dest[i] = a[i] * b[i];
}
} else {
for (unsigned int i = idx; i < numEls; i += numThreads) {
dest[i] = scale_targets * dest[i] + a[i] * b[i];
}
}
}