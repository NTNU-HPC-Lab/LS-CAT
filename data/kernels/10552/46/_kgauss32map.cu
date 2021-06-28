#include "includes.h"
__global__ void _kgauss32map(int nx, int ns, float *x2, float *s2, float *k, float g) {
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