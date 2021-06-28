#include "includes.h"
__global__ void sumArraysOnGPU(double *A, double *B, double *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N) C[i] = A[i] + B[i] + 7*A[i] + 4*B[i]/123.1 - B[i]*A[i] + B[i]*B[i] - 9*B[i]*B[i]*B[i]/0.4 + A[i]/0.2 + B[i]*B[i];
}