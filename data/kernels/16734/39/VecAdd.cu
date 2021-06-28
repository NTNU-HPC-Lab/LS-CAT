#include "includes.h"
__global__ void VecAdd(const int* A, const int* B, int* C, int size)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
for(int n = 0 ; n < 100; n++) {
C[i] += A[i] + B[i];
}
}