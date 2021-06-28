#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx<N)
{
C[idx] = A[idx] + B[idx];
}
}