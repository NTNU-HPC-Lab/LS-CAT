#include "includes.h"
__global__ void transpose_read_row_write_column(int * mat, int * transpose, int nx, int ny)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

if (ix < nx && iy < ny)
{
transpose[ix * ny + iy] = mat[iy * nx + ix];
}
}