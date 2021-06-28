#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
//int i = threadIdx.x;
int i = blockIdx.x * blockDim.x + threadIdx.x; //general case
if (i < N) C[i] = B[i] + A[i];
}