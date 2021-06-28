#include "includes.h"
__global__ void saxpy_kernel(const float a, const float* x, const float* y, float* result, unsigned int len) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < len) result[idx] = a * x[idx] + y[idx];
}