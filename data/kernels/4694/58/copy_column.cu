#include "includes.h"
__global__ void copy_column(int * mat, int * transpose, int nx, int ny)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

if (ix < nx && iy < ny)
{
transpose[ix * ny + iy] = mat[ix * ny + iy];
}
}