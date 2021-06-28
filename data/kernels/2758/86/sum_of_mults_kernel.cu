#include "includes.h"
__global__ void sum_of_mults_kernel(float *a1, float *a2, float *b1, float *b2, size_t size, float *dst)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index < size) {
dst[index] = a1[index] * a2[index] + b1[index] * b2[index];
}
}