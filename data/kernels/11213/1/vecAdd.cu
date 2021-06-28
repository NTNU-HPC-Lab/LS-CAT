#include "includes.h"
__global__ void vecAdd(float* A, float* B, float* C) {
//threadIdx.x is a build-in variable provided by CUDA runtime
int i = threadIdx.x;
A[i] = 0;
B[i] = 0;
C[i] = A[i] + B[i];
}