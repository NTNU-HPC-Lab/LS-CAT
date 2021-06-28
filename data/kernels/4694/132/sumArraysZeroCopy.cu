#include "includes.h"
__global__ void sumArraysZeroCopy(int *A, int *B, int *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i];
}