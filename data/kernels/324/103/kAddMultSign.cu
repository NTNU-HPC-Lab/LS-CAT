#include "includes.h"
__global__ void kAddMultSign(float* a, float* b, unsigned int numEls, float mult) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < numEls; i += numThreads) {
a[i] = a[i] + ((b[i] > 0) ? mult : ((b[i] < 0) ? -mult : 0));
}
}