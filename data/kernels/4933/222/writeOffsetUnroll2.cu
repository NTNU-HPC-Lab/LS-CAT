#include "includes.h"
__global__ void writeOffsetUnroll2(float *A, float *B, float *C, const int n, int offset)
{
unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int k = i + offset;

if (k + blockDim.x < n)
{
C[k]            = A[i]            + B[i];
C[k + blockDim.x] = A[i + blockDim.x] + B[i + blockDim.x];
}
}