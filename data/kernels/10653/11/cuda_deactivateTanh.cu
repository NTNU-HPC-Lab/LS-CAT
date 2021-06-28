#include "includes.h"
__global__ void cuda_deactivateTanh(double* pE, const double* pA, int n)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < n) {
pE[id] *= (1.0 - (pA[id] * pA[id]));
}
}