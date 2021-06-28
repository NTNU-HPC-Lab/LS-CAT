#include "includes.h"
__global__ void transposeNaiveCol(int *in, int *out, const int nx, const int ny)
{
// set thread id.
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

if (ix < nx && iy < ny)
{
out[iy * nx + ix] = in[ix * ny + iy];
}
}