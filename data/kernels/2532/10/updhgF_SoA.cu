#include "includes.h"
__global__ void updhgF_SoA(float *f, float *z1, float *z2, float *g, float tf, float invlambda, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;
float DIVZ;

if (px<nx && py<ny)
{
// compute the divergence
DIVZ = 0;
if ((px<(nx - 1))) DIVZ += z1[idx];
if ((px>0))      DIVZ -= z1[idx - 1];

if ((py<(ny - 1))) DIVZ += z2[idx];
if ((py>0))      DIVZ -= z2[idx - nx];

// update f
f[idx] = (1 - tf) *f[idx] + tf * (g[idx] + invlambda*DIVZ);
}

}