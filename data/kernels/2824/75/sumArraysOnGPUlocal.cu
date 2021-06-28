#include "includes.h"
__global__ void sumArraysOnGPUlocal(float *A, float *B, float *C, const int N)
{
float local[4];
int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i +4 < N) {
for (int j=0; j < 4; j++) local[j] = 2*A[i+j];
C[i] = A[i] + B[i] + local[threadIdx.x%4];

}

}