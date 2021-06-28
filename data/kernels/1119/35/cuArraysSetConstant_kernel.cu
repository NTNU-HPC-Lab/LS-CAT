#include "includes.h"
__global__ void cuArraysSetConstant_kernel(float *image, int size, float value)
{
int idx = threadIdx.x + blockDim.x*blockIdx.x;

if(idx < size)
{
image[idx] = value;
}
}