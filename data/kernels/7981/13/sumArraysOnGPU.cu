#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N){
int i = blockDim.x * blockIdx.x + threadIdx.x;
if (i < N)
C[i] = A[i] + B[i];
}