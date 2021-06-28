#include "includes.h"
__global__ void transposeDiagonalCol(float *out, float *in, const int nx, const int ny)
{
unsigned int blk_y = blockIdx.x;
unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

unsigned int ix = blockDim.x * blk_x + threadIdx.x;
unsigned int iy = blockDim.y * blk_y + threadIdx.y;

if (ix < nx && iy < ny)
{
out[iy * nx + ix] = in[ix * ny + iy];
}
}