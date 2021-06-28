#include "includes.h"
__global__ void VecAdd(const int* A, const int* B, int* C, int N) {

// Index holen
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < N)
C[i] = A[i] + B[i];

}