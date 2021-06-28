#include "includes.h"
__global__ void matrixAdd(float *A, float *B, float *C, int n)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

int num = n;

int i = row * num + col;

if (row < num && col < num)
{
C[i] = A[i] + B[i];
}
}