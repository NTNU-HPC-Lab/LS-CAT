#include "includes.h"
__global__ void pnpoly_cnGPU2(const float *px, const float *py, const float *vx, const float *vy, char* cs, int npoint, int nvert)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
__shared__ float tpx;
__shared__ float tpy;
if (i < npoint) {
tpx = px[i];
tpy = py[i];
int j, k, c = 0;
for (j = 0, k = nvert-1; j < nvert; k = j++) {
if ( ((vy[j]>tpy) != (vy[k]>tpy)) &&
(tpx < (vx[k]-vx[j]) * (tpy-vy[j]) / (vy[k]-vy[j]) + vx[j]) )
c = !c;
}
cs[i] = c & 1;
__syncthreads();
}
}