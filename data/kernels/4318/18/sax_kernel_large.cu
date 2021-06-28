#include "includes.h"
__global__ void sax_kernel_large(const float a, const float* x, float* result, unsigned int len, unsigned int rowsz) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * rowsz;
if (idx < len) result[idx] = a * x[idx];
}