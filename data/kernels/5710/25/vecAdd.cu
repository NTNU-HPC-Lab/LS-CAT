#include "includes.h"
__global__ void vecAdd(float *A, float *B, float *C) {
int i;

i = blockIdx.x*blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];

}