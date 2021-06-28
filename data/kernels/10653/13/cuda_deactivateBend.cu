#include "includes.h"
__global__ void cuda_deactivateBend(double* pE, const double* pA, int n)
{
int id = blockIdx.x * blockDim.x + threadIdx.x;
if (id < n) {
double x = pE[id];
pE[id] *= 0.5 * (x / sqrt(x * x + 1)) + 1;
}
}