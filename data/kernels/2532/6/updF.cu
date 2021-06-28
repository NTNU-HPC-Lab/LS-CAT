#include "includes.h"
__global__ void updF(float *f, float *z, float *g, float tf, float lambda, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;
float DIVZ;

if (px<nx && py<ny)
{
// compute the divergence
DIVZ = 0;
if ((px<(nx - 1))) DIVZ += z[2 * (idx)+0];
if ((px>0))      DIVZ -= z[2 * (idx - 1) + 0];

if ((py<(ny - 1))) DIVZ += z[2 * (idx)+1];
if ((py>0))      DIVZ -= z[2 * (idx - nx) + 1];

// update f
//f[idx] = (1.-tf*lambda)*f[idx] + tf * DIVZ + tf*lambda*g[idx];
f[idx] = (f[idx] + tf * DIVZ + tf*lambda*g[idx]) / (1 + tf*lambda);
}

}