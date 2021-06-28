#include "includes.h"
__global__ void sub0(float *div0, float *div, float *g, float lambda, int nx, int ny)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = x + y*nx;
if (x<nx && y<ny)   div[idx] = div0[idx] - g[idx] / lambda;
}