#include "includes.h"
__global__ void vecAdd(int *A, int *B, int *C) {
int i = blockIdx.x*blockDim.x+threadIdx.x;
C[i] = A[i];
}