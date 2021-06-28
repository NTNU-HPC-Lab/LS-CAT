#include "includes.h"
__global__ void sumArraysZeroCopyWithUVAOffset(float *A, float *B, float *C, const int N, int offset)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i + offset] = A[i + offset] + B[i + offset];
}