#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
float d[16];

if (i < N) {
d[threadIdx.x%16]= A[i] + B[i];
C[i] = d[threadIdx.x%8];
}
}