#include "includes.h"
__global__ void interleave(float* input, float* output, int size) {
const int numThreads = blockDim.x * gridDim.x;
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = threadID; i < size; i += numThreads) {
output[2 * i] = input[i];
output[2 * i + 1] = input[size + 2 + i];
}
}