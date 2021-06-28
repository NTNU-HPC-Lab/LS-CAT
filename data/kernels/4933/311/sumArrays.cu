#include "includes.h"
__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < N)
{
for (int i = 0; i < N; ++i)
{
C[idx] = A[idx] + B[idx];
}
}
}