#include "includes.h"
__global__ void MyFloatScale(float *a, float scale, int size) {
const int numThreads = blockDim.x * gridDim.x;
const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = threadID; i < size; i += numThreads)
{
a[i] *= scale;
}
}