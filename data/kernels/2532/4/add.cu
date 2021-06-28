#include "includes.h"
__global__ void add(float *a, float *b, int nx, int ny)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = x + y*nx;
if (x<nx && y<ny)   b[idx] += a[idx] * .125;

}