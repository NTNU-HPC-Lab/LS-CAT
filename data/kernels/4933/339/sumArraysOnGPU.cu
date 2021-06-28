#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
}