#include "includes.h"


__global__ void VecAdd(float * A, float * B, float * C)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
}