#include "includes.h"
__global__ void BackwardReLU(float* Z, float* dA, int nRowsdZ, int nColsdZ, float *dZ)
{
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < nRowsdZ * nColsdZ)
{
if (Z[index] >= 0)
dZ[index] = dA[index];
else
dZ[index] = 0;
}
}