#include "includes.h"
__global__ void kernelGetOmega(const int N, double *omega, double *kSqr, const double sigma2, const double sigma4, const double lambda, const double g)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N)
{
omega[i] = sqrt(1 + kSqr[i] + 3 * lambda * sigma2 + 15 * g * sigma4);
}
}