#include "includes.h"
__global__ void FloatDivByConstant(float *A, float constant)
{
unsigned int i = blockIdx.x * gridDim.y * gridDim.z * blockDim.x + blockIdx.y * gridDim.z * blockDim.x + blockIdx.z * blockDim.x + threadIdx.x;
A[i]=A[i]/constant;
}