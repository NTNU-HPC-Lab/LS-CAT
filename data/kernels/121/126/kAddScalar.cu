#include "includes.h"
__global__ void kAddScalar(float* a, float alpha, float* dest, unsigned int numEls) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;
for (unsigned int i = idx; i < numEls; i += numThreads) {
dest[i] = a[i] + alpha;
}
}