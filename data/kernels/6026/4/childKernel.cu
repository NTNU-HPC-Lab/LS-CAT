#include "includes.h"
__global__ void childKernel(unsigned int parentThreadIndex, float* data) {
data[threadIdx.x] = parentThreadIndex + 0.1f * threadIdx.x;
}