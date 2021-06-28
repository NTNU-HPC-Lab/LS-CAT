#include "includes.h"
__global__ void UpdateParamsLinear(float *dZ, float *A, int nRowsdZ, int nColsdZ, int nRowsA, float lr, float *W, float *b)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

float dWValue = 0, dbValue = 0;

if (row < nRowsdZ && col < nRowsA)
{
for (int i = 0; i < nColsdZ; i++)
{
dWValue += dZ[row * nColsdZ + i] * A[col * nColsdZ + i];
}
W[row * nRowsA + col] = W[row * nRowsA + col] - lr * dWValue / nColsdZ;

if (col == 0)
{
for (int i = 0; i < nColsdZ; i++)
{
dbValue += dZ[row * nColsdZ + i];
}
b[row] = b[row] - lr * dbValue / nColsdZ;
}
}
}