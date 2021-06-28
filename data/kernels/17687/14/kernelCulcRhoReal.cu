#include "includes.h"
__global__ void kernelCulcRhoReal(const int N, double *rho, double *q, double *p, const double lambda, const double g)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N)
{
double qi = q[i];
double pi = p[i];

rho[i] = 0.5 * qi * qi;
rho[i] += 0.5 * pi * pi;
rho[i] += (lambda / 4.0) * qi * qi * qi * qi;
rho[i] += (g / 6.0)  * qi * qi * qi * qi * qi * qi;
}
}