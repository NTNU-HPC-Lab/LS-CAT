#include "includes.h"
__global__ void matrixMultKernel (float *d_A, float *d_B, float *d_C, int N)
{
// Calculate the row index of the d_C element and d_A
int row = blockIdx.y * blockDim.y + threadIdx.y;

// Calculate the column index of d_C and d_B
int col = blockIdx.x * blockDim.x + threadIdx.x;

if ((row < N) && (col < N))
{
float Cvalue = 0;
for (int k = 0; k < N; k++)
Cvalue += d_A[row * N + k] * d_B[k * N + col];
d_C[row * N + col] = Cvalue;
}
}