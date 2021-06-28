#include "includes.h"
__global__ void lap(float *a, float *b, int nx, int ny)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = x + y*nx;

if (x<nx && y<ny)
{
float AX = 0, BX = 0;
if (x>0)   { BX += a[idx - 1]; AX++; }
if (y>0)   { BX += a[idx - nx]; AX++; }
if (x<nx - 1){ BX += a[idx + 1]; AX++; }
if (y<ny - 1){ BX += a[idx + nx]; AX++; }
b[idx] = -AX*a[idx] + BX;
}
}