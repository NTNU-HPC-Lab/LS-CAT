#include "includes.h"
__global__ void matrixMul_kernel(float * A, float * B, float * C, int N)
{
int ROW = blockIdx.y * blockDim.y + threadIdx.y;
int COL = blockIdx.x * blockDim.x + threadIdx.x;

float tmpSum = 0;

if (ROW < N && COL < N)
{
// each thread computes one elem of the block sub-matrix
for (int i = 0; i < N; i++)
{
tmpSum += A[ROW * N + i] * B[i * N + COL];
}
}
C[ROW * N + COL] = tmpSum;
}