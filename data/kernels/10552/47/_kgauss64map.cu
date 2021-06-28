#include "includes.h"
__global__ void _kgauss64map(int nx, int ns, double *x2, double *s2, double *k, double g) {
int i, n, xi, si;
i = threadIdx.x + blockIdx.x * blockDim.x;
n = nx*ns;
while (i < n) {
xi = (i % nx);
si = (i / nx);
k[i] = exp(-g * (x2[xi] + s2[si] - 2*k[i]));
i += blockDim.x * gridDim.x;
}
}