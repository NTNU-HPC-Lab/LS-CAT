#include "includes.h"
__global__ void naiveGmemUnroll(float *out, float *in, const int nx, const int ny)
{
unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

unsigned int ti = iy * nx + ix;
unsigned int to = ix * ny + iy;

if (ix + blockDim.x < nx && iy < ny)
{
out[to]                   = in[ti];
out[to + ny * blockDim.x]   = in[ti + blockDim.x];
}
}