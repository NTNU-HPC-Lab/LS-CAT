#include "includes.h"
__global__ void sum10ops(float *A, float *B, float *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < N) C[i] = A[i] + B[i] - A[i]*A[i] + 3*B[i] - 4*A[i]*B[i] + B[i]*B[i]*7- 8;
}