#include "includes.h"

#define BLOCK_SIZE 32
#define N 2048

__global__ void matMult(float* A, float* B, float* C){
// Индекс блока
int bx = blockIdx.x;
int by = blockIdx.y;

// Индекс нити
int tx = threadIdx.x;
int ty = threadIdx.y;

float sum = 0.0;
//Индекс A[i][0]
int ia = N * BLOCK_SIZE * by + N * ty;
// Индекс B[0][j]
int ib = BLOCK_SIZE * bx + tx;


for (int k = 0; k < N; k++) {
sum += A[ia + k] * B[ib + k * N];
}
// Индекс C[i][j]
int ic = N * BLOCK_SIZE * by + BLOCK_SIZE * bx;

//Результирующая матрица
C[ic + N * ty + tx] = sum;
}