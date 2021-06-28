#include "includes.h"
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny)
{
unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

if (ix < nx && iy < ny)
{
out[ix * ny + iy] = in[iy * nx + ix];
}
}