#include "includes.h"

#define BLOCK_SIZE 100
#define GRID_SIZE 100
#define N GRID_SIZE * BLOCK_SIZE


__global__ void VectorAdd (int *A, int *B, int *C) {
int x = threadIdx.x + blockIdx.x * blockDim.x;
C[x] = A[x] + B[x];
}