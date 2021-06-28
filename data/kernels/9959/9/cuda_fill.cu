#include "includes.h"
__global__ void cuda_fill(double* pVec, double val, int n)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < n)
pVec[n] = val;
}