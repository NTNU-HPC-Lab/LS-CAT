#include "includes.h"
__global__ void transpose_diagonal_row(int * mat, int * transpose, int nx, int ny)
{
int blk_x = blockIdx.x;
int blk_y = (blockIdx.x + blockIdx.y) % gridDim.x;

int ix = blockIdx.x * blk_x + threadIdx.x;
int iy = blockIdx.y * blk_y + threadIdx.y;

if (ix < nx && iy < ny)
{
transpose[ix * ny + iy] = mat[iy * nx + ix];
}
}