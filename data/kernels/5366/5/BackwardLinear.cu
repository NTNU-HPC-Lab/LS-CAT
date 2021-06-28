#include "includes.h"
__global__ void BackwardLinear(float *dZ, float *W, int nColsW, int nRowsW, int nColsdZ, float *dA)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float dAValue = 0;

if (row < nColsW && col < nColsdZ)
{
for (int i = 0; i < nRowsW; i++)
{
dAValue += W[i * nColsW + row] * dZ[i * nColsdZ + col];
}
dA[row * nColsdZ + col] = dAValue;
}
}