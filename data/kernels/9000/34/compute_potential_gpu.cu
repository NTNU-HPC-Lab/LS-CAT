#include "includes.h"
__global__ void compute_potential_gpu(float *m, float *x, float *y, float *z, float *phi, int N, int N1) {
int i,j;
float rijx, rijy, rijz;
float xi, yi, zi;
float potential;
i = threadIdx.x + blockIdx.x*blockDim.x;
if (i < (N1 == 0 ? N : N1))
{
xi = x[i];
yi = y[i];
zi = z[i];

for (j = (N1 == 0 ? 0 : N1); j < N; j++)
{
rijx = xi - x[j];
rijy = yi - y[j];
rijz = zi - z[j];

if (i!=j)
potential -= m[j]/sqrt(rijx*rijx + rijy*rijy + rijz*rijz);
}
phi[i] = potential;
}
}