#include "includes.h"
__global__ void ForwardLinear(float *A, float *W, float *b, int nRowsW, int nColsW, int nColsA, float *Z)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float ZValue = 0;

if (row < nRowsW && col < nColsA)
{
for (int i = 0; i < nColsW; i++)
{
ZValue += W[row * nColsW + i] * A[i * nColsA + col];
}
Z[row * nColsA + col] = ZValue + b[row];
}
}