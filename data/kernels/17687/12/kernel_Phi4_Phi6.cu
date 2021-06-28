#include "includes.h"
__global__ void kernel_Phi4_Phi6(const int N, double *t, double *q, const double lambda, const double g)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N)
{
t[i] = q[i] * q[i] * q[i] * (lambda + g * q[i] * q[i]);
}
}