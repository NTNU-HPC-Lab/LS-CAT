#include "includes.h"
// ïîäêëþ÷åíèå áèáëèîòåêè cuBLAS


#define IDX2C(i,j,ld) (((i)*(ld))+(j))


__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N)
{
int ROW = blockIdx.y*blockDim.y + threadIdx.y;
int COL = blockIdx.x*blockDim.x + threadIdx.x;
float tmpSum = 0;

if (ROW < N && COL < N) {
// each thread computes one element of the block sub-matrix
for (int i = 0; i < N; i++) {
tmpSum += A[ROW * N + i] * B[i * N + COL];
}

C[ROW * N + COL] = tmpSum;
}
}