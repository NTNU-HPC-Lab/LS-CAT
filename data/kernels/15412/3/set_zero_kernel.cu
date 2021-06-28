#include "includes.h"
__global__ void set_zero_kernel(float *src, int size)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < size) src[i] = 0;
}