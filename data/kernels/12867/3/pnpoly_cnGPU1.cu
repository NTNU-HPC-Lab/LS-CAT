#include "includes.h"
__global__ void pnpoly_cnGPU1(const float *px, const float *py, const float *vx, const float *vy, char* cs, int npoint, int nvert)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < npoint) {
int j, k, c = 0;
for (j = 0, k = nvert-1; j < nvert; k = j++) {
if ( ((vy[j]>py[i]) != (vy[k]>py[i])) &&
(px[i] < (vx[k]-vx[j]) * (py[i]-vy[j]) / (vy[k]-vy[j]) + vx[j]) )
c = !c;
}
cs[i] = c & 1;
}
}