#include "includes.h"
__global__ void iKernel(float *src, float *dst)
{
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
dst[idx] = src[idx] * 2.0f;
}