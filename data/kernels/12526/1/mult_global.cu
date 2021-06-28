#include "includes.h"
__global__ void mult_global (int *A, int *B, int *result, int n)
{
int k, sum = 0;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

if(col < n && row  < n)
{
for (k = 0; k < n; k++)
{
sum += A[row * n + k] * B[k * n + col];
result[row * n + col] = sum;
}
}
}