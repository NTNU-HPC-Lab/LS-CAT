#include "includes.h"
__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
int id = threadIdx.x;
C[id] = A[id] + B[id];
}