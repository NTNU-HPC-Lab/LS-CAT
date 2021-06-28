#include "includes.h"
__global__ void VecAdd(const float* A, const float* B, float* C)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
C[i] = A[i] + B[i];
}