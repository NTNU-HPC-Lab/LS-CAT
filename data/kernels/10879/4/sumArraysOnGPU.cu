#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C)
{
int i = threadIdx.x;
C[i] = A[i] + B[i];
}