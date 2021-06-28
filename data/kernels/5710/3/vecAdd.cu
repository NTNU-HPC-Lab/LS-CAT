#include "includes.h"
__global__ void vecAdd(int *A, int *B, int *C) {
int i = threadIdx.x;
C[i] = A[i] + B[i];
}