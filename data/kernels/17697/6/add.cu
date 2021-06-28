#include "includes.h"
extern "C" {
}
__global__ void add(const float* x1, const float* x2, float* y, unsigned int len) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if (tid < len) {
y[tid] = x1[tid] + x2[tid];
}
}