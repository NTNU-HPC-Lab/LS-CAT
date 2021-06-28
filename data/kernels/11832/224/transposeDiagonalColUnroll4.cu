#include "includes.h"
__global__ void transposeDiagonalColUnroll4(float *out, float *in, const int nx, const int ny)
{
unsigned int blk_y = blockIdx.x;
unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

unsigned int ix_stride = blockDim.x * blk_x;
unsigned int ix = ix_stride * 4 + threadIdx.x;
unsigned int iy = blockDim.y * blk_y + threadIdx.y;

if (ix < nx && iy < ny)
{
out[iy * nx + ix] = in[ix * ny + iy];
out[iy * nx + ix + blockDim.x] = in[(ix + blockDim.x) * ny + iy];
out[iy * nx + ix + 2 * blockDim.x] =
in[(ix + 2 * blockDim.x) * ny + iy];
out[iy * nx + ix + 3 * blockDim.x] =
in[(ix + 3 * blockDim.x) * ny + iy];
}
}