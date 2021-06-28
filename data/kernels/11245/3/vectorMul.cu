#include "includes.h"
__global__ void vectorMul(const float *A, const float *B, float *C, int numElements)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;


if (i < numElements)
{
C[i] = A[i] * B[i];
}
}