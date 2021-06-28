#include "includes.h"
__global__ void _kgauss64d(int nx, int ns, int nd, double *x, double *s, double *k, double g) {
int i, j, n, xj, sj;
double d, dd;
i = threadIdx.x + blockIdx.x * blockDim.x;
n = nx*ns;
while (i < n) {
xj = (i % nx)*nd;
sj = (i / nx)*nd;
dd = 0;
for (j = 0; j < nd; j++) {
d = x[xj++]-s[sj++];
dd += d*d;
}
k[i] = exp(-g * dd);
i += blockDim.x * gridDim.x;
}
}