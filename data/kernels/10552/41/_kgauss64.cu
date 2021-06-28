#include "includes.h"
__global__ void _kgauss64(int nx, int ns, double *xval, int *xrow, int *xcol, double *sval, int *srow, int *scol, double *k, double g) {
int i, n, x1, x2, xc, xr, s1, s2, sc, sr;
double d, dd;
i = threadIdx.x + blockIdx.x * blockDim.x;
n = nx*ns;
while (i < n) {
xc = i % nx;
sc = i / nx;
x1 = xcol[xc]-1; x2 = xcol[xc+1]-1;
s1 = scol[sc]-1; s2 = scol[sc+1]-1;
dd = 0;
while ((x1 < x2) || (s1 < s2)) {
xr = ((x1 < x2) ? xrow[x1] : INT_MAX);
sr = ((s1 < s2) ? srow[s1] : INT_MAX);
d = ((sr < xr) ? sval[s1++] :
(xr < sr) ? xval[x1++] :
(xval[x1++]-sval[s1++]));
dd += d*d;
}
k[i] = exp(-g * dd);
i += blockDim.x * gridDim.x;
}
}