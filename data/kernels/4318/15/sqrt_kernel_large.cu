#include "includes.h"
__global__ void sqrt_kernel_large(float* x, unsigned int len, unsigned int rowsz) {
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
if (idx < len) x[idx] = sqrt(x[idx]);
}