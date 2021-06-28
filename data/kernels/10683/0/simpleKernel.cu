#include "includes.h"
__global__ void simpleKernel(float *dst, float *src1, float *src2)
{
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//float temp = src[idx];
dst[idx] = src1[idx] + src2[idx];
}