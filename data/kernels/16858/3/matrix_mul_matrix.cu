#include "includes.h"
__global__ void matrix_mul_matrix(float *A, float *B, float *C, int col_A, int col_B, int row_C, int col_C){
float sum = 0.0f;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < row_C && col < col_C) {
for (int i = 0; i < col_A; ++i) {
sum += A[row * col_A + i] * B[i * col_B + col];
}
C[row * col_B + col] = sum;
}
}