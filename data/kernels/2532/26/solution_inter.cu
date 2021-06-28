#include "includes.h"
__global__ void solution_inter(float *z, float *g, float lambda, int nx, int ny)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = x + y*nx;

if (x<nx && y<ny)   g[idx] = -z[3 * idx + 2] * lambda;
}