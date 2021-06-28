#include "includes.h"
__global__ void vxy_kernel(const float* x, float* y, float* result, unsigned int len) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) result[idx] = x[idx] * y[idx];
}