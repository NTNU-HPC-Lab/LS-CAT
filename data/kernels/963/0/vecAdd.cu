#include "includes.h"



#define N 1000	// size of vectors

#define T 10000// number of threads per block


__global__ void vecAdd(int *A, int *B, int *C) {
int i = blockIdx.x*blockDim.x + threadIdx.x;
C[i] = A[i] * 10 + B[i];
}