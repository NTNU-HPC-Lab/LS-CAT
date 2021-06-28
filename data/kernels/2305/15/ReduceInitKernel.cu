#include "includes.h"



__global__ void ReduceInitKernel(float *dst, int length)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;

if (x < length) {
dst[x] = 0;
}
}