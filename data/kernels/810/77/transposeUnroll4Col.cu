#include "includes.h"
__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny)
{
unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

unsigned int ti = iy * nx + ix; // access in rows
unsigned int to = ix * ny + iy; // access in columns

if (ix + 3 * blockDim.x < nx && iy < ny)
{
out[ti]                = in[to];
out[ti +   blockDim.x] = in[to +   blockDim.x * ny];
out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
}
}