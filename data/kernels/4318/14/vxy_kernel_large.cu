#include "includes.h"
__global__ void vxy_kernel_large(const float* x, float* y, float* result, unsigned int len, unsigned int rowsz) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + rowsz * blockIdx.y;
if (idx < len) result[idx] = x[idx] * y[idx];
}