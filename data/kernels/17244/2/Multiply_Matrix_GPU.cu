#include "includes.h"
__global__ void Multiply_Matrix_GPU(float* A, float* B, float* C , int BLOCK_SIZE , int N) {
// Индекс блока
int bx = blockIdx.x;
int by = blockIdx.y;

// Индекс нити
int tx = threadIdx.x;
int ty = threadIdx.y;

float total = 0.0;
int ia = N * BLOCK_SIZE * by + N * ty;
int ib = BLOCK_SIZE * bx + tx;

for (int k = 0; k < N; k++) {
total += A[ia + k] * B[ib + k * N];
}
int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;

//Результирующая матрица
C[ic + N * ty + tx] = total;
}