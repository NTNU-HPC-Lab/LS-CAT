#include "includes.h"
__global__ void transpose_unroll4_row(int * mat, int * transpose, int nx, int ny)
{
int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

int ti = iy * nx + ix;
int to = ix * ny + iy;

if (ix + 3 * blockDim.x < nx && iy < ny)
{
transpose[to]						= mat[ti];
transpose[to + ny*blockDim.x]		= mat[ti + blockDim.x];
transpose[to + ny * 2 * blockDim.x] = mat[ti + 2 * blockDim.x];
transpose[to + ny * 3 * blockDim.x] = mat[ti + 3 * blockDim.x];
}
}