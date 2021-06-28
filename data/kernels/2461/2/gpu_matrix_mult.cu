#include "includes.h"
__global__ void gpu_matrix_mult(float *A, float *B, float *C,  int n)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < n && col < n) {
for (int i = 0; i < n; ++i) {
C[row * n + col] += A[row * n + i] * B[i * n + col];
}
}
}