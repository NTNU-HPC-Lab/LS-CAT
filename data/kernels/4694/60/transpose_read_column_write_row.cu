#include "includes.h"
__global__ void transpose_read_column_write_row(int * mat, int * transpose, int nx, int ny)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

if (ix < nx && iy < ny)
{
transpose[iy * nx + ix] = mat[ix * ny + iy];
}
}