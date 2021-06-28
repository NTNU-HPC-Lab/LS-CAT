#include "includes.h"

__global__ void vec_add(int* A, int* B, int* C, int size) {

int index = threadIdx.x + blockIdx.x * blockDim.x;
if(index < size) {
C[index] = A[index] + B[index];
}
}