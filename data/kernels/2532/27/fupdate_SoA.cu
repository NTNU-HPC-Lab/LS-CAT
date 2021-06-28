#include "includes.h"
__global__ void fupdate_SoA(float *f, float *z1, float *z2, float *g, float invlambda, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;
float DIVZ;

if (px<nx && py<ny)
{
// compute the divergence
DIVZ = 0;
float Z1c = z1[(idx)];
float Z2c = z2[(idx)];
float Z1l = z1[(idx - 1)];
float Z2d = z2[(idx - nx)];
if (!(px == (nx - 1))) DIVZ += Z1c;
if (!(px == 0))      DIVZ -= Z1l;
if (!(py == (ny - 1))) DIVZ += Z2c;
if (!(py == 0))      DIVZ -= Z2d;

// update f
f[idx] = DIVZ - g[idx] * invlambda;
}
}