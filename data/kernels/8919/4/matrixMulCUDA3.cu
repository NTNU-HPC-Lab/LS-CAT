#include "includes.h"
__global__ void matrixMulCUDA3(float *C, float  *B, float *A, int n)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x+ threadIdx.x;

float sum = 0.0f;

if (row >= n || col >= n) {
return;
}

for (int k = 0; k < n; k++) {
sum += A[row * n + k] * B[k * n + col];
}
C[row * n + col] = sum;
}