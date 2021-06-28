#include "includes.h"
__global__ void gradient(float *u, float *g, int nx, int ny)
{
int px = blockIdx.x * blockDim.x + threadIdx.x;
int py = blockIdx.y * blockDim.y + threadIdx.y;
int idx = px + py*nx;
/*
if  (idx<N)
{
g[2*idx+0] = 0;
g[2*idx+1] = 0;
}
if ((idx< N) && px<(nx-1)) g[2*idx+0] = u[idx+1 ] - u[idx];
if ((idx< N) && py<(ny-1)) g[2*idx+1] = u[idx+nx] - u[idx];
*/
if (px<nx && py<ny)
{
g[2 * idx + 0] = 0;
g[2 * idx + 1] = 0;
if (px<(nx - 1)) g[2 * idx + 0] = u[idx + 1] - u[idx];
if (py<(ny - 1)) g[2 * idx + 1] = u[idx + nx] - u[idx];
}
//a[idx] =0;
}