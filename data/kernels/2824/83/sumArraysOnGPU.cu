#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = 2*A[i] + 3*B[i] - A[i] - 2*B[i];
}