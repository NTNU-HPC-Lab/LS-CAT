#include "includes.h"



__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
int idx =  blockIdx.x * blockDim.x + threadIdx.x;
C[idx] = A[idx] + B[idx];
}