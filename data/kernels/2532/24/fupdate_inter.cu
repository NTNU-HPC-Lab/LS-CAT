#include "includes.h"
__global__ void fupdate_inter(float *z, float *g, float invlambda, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;
float DIVZ;

if (px<nx && py<ny)
{
// compute the divergence
DIVZ = 0;
if ((px<(nx - 1))) DIVZ += z[3 * (idx)+0];
if ((px>0))      DIVZ -= z[3 * (idx - 1) + 0];

if ((py<(ny - 1))) DIVZ += z[3 * (idx)+1];
if ((py>0))      DIVZ -= z[3 * (idx - nx) + 1];

// update f
z[3 * idx + 2] = DIVZ - g[idx] * invlambda;
}
}