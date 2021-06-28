#include "includes.h"
__global__ void fillZeros(float* buf, int size) {
const int numThreads = blockDim.x * gridDim.x;
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = threadID; i < size; i += numThreads)
{
buf[i] = 0.0f;
}
}