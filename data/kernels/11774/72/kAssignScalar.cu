#include "includes.h"
__global__ void kAssignScalar(float* dest, float alpha, unsigned int len) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < len; i += numThreads) {
dest[i] = alpha;
}
}