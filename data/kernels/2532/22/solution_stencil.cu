#include "includes.h"
__global__ void solution_stencil(float *zx, float * zy, float *g, float lambda, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;

float DIVZ;

if (px<nx && py<ny)
{
// compute the divergence
DIVZ = 0;
if ((px<(nx - 1))) DIVZ += zx[(idx)];
if ((px>0))      DIVZ -= zx[(idx - 1)];

if ((py<(ny - 1))) DIVZ += zy[(idx)];
if ((py>0))      DIVZ -= zy[(idx - nx)];

// update f
g[idx] = -DIVZ*lambda + g[idx];
}
}