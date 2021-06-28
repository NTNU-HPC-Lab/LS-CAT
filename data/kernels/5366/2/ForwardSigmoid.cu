#include "includes.h"
__global__ void ForwardSigmoid(float* Z, int nRowsZ, int nColsZ, float* A)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < nRowsZ * nColsZ)
{
A[index] = 1 / (1 + exp(-Z[index]));
}
}