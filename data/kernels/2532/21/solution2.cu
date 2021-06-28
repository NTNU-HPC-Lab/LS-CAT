#include "includes.h"
__global__ void solution2(float *f, float lambda, int nx, int ny)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = x + y*nx;


if (x<nx && y<ny)   f[idx] = -f[idx] * lambda;
}