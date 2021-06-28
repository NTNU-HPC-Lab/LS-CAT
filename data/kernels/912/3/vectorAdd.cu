#include "includes.h"
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < n)
{
C[i] = A[i] + B[i];
}
}