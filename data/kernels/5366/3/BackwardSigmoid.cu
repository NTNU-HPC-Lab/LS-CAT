#include "includes.h"
__global__ void BackwardSigmoid(float* Z, float* dA, int nRowsdZ, int nColsdZ, float *dZ)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < nRowsdZ * nColsdZ)
{
dZ[index] = 1 / (1 + exp(-Z[index])) * (1 - 1 / (1 + exp(-Z[index]))) *
dA[index];
}
}