#include "includes.h"
__global__ void _A_mul_Bst_32(int my, int xc, float *dy, float *xval, int *xrow, int *xcol, float *dw) {
// dw[wr,wc] += dy[yr,yc] * x[xr,xc]  where wr=yr, wc=xr, yc=xc
int t, n, xp, xr, yp, yr, wp;
t = threadIdx.x + blockIdx.x * blockDim.x;
n = xcol[xc+1] - xcol[xc];
while (t < n) {
xp = xcol[xc] + t - 1;
xr = xrow[xp] - 1;
for (yr = 0; yr < my; yr++) {
yp = yr + xc * my;
wp = yr + xr * my;
dw[wp] += dy[yp] * xval[xp];
}
t += blockDim.x * gridDim.x;
}
}