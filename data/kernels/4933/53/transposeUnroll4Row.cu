#include "includes.h"
__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny)
{
unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

unsigned int ti = iy * nx + ix; // access in rows
unsigned int to = ix * ny + iy; // access in columns

if (ix + 3 * blockDim.x < nx && iy < ny)
{
out[to]                   = in[ti];
out[to + ny * blockDim.x]   = in[ti + blockDim.x];
out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
}
}