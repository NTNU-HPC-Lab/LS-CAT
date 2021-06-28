#include "includes.h"
__global__ void rsqrt_kernel_large(float* x, unsigned int len, unsigned int rowsz) {
unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * rowsz;
if (idx < len) x[idx] = x[idx] > 0 ? rsqrt(x[idx]) : 0;
}