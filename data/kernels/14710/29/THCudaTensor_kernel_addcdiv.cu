#include "includes.h"
__global__ void THCudaTensor_kernel_addcdiv(float *data, float value, float *src1, float *src2, long size)
{
long k = (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;

if(k < size)
data[k] += value*src1[k]/src2[k];
}