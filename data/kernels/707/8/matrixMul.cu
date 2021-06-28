#include "includes.h"
__global__ void matrixMul(int *A, int *B, int *C, int n)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

int num = n;

if (row < num && col < num)
{
long Cvalue = 0;
for (int i = 0; i < num; i++)
{
Cvalue += A[row * num + i] * B[i * num + col];
}
C[row * num + col] = Cvalue;
}
}