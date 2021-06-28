#include "includes.h"
__global__ void simpleKernel(float *dst, float *src)
{
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
float temp = src[idx];
dst[idx] = temp * temp;
}