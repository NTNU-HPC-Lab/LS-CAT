#include "includes.h"
__global__ void pnpoly_cnGPU(char *cs, const float *px, const float *py, const float *vx, const float *vy, int npoint, int nvert)
{
extern __shared__ int s[];
float *tvx = (float*) s;
float *tvy = (float*)&s[nvert];

int i = blockIdx.x*blockDim.x + threadIdx.x;
if (i < npoint) {
int j, k, c = 0;
for (j = 0, k = nvert-1; j < nvert; k = j++) {
tvx[j] = vx [j];
tvy[j] = vy [j];
}

__syncthreads();

for (j = 0, k = nvert-1; j < nvert; k = j++) {
if ( ((tvy[j]>py[i]) != (tvy[k]>py[i])) &&
(px[i] < (tvx[k]-tvx[j]) * (py[i]-tvy[j]) / (tvy[k]-tvy[j]) + tvx[j]) )
c = !c;
}
cs[i] = c & 1;
}
}