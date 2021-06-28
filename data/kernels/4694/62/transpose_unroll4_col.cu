#include "includes.h"
__global__ void transpose_unroll4_col(int * mat, int * transpose, int nx, int ny)
{
int ix = blockIdx.x * blockDim.x * 4 + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

int ti = iy * nx + ix;
int to = ix * ny + iy;

if (ix + 3 * blockDim.x < nx && iy < ny)
{
transpose[ti] = mat[to];
transpose[ti + blockDim.x] = mat[to + blockDim.x*ny];
transpose[ti + 2 * blockDim.x] = mat[to + 2 * blockDim.x*ny];
transpose[ti + 3 * blockDim.x] = mat[to + 3 * blockDim.x*ny];
}
}