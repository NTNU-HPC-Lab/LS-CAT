#include "includes.h"
__global__ void ForwardReLU(float* Z, int nRowsZ, int nColsZ, float* A)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < nRowsZ * nColsZ)
{
if (Z[index] >= 0)
A[index] = Z[index];
else
A[index] = 0;
}
}