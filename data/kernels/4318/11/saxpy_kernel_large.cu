#include "includes.h"
__global__ void saxpy_kernel_large(const float a, const float* x, const float* y, float* result, unsigned int len, unsigned int rowsz) {
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
if (idx < len) result[idx] = a * x[idx] + y[idx];
}