#include "includes.h"
__global__ void transposeUnroll8Row(float *out, float *in, const int nx, const int ny)
{
unsigned int ix = blockDim.x * blockIdx.x * 8 + threadIdx.x;
unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

unsigned int ti = iy * nx + ix; // access in rows
unsigned int to = ix * ny + iy; // access in columns

if (ix + 7 * blockDim.x < nx && iy < ny)
{
out[to]                   = in[ti];
out[to + ny * blockDim.x]   = in[ti + blockDim.x];
out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
out[to + ny * 4 * blockDim.x] = in[ti + 4 * blockDim.x];
out[to + ny * 5 * blockDim.x] = in[ti + 5 * blockDim.x];
out[to + ny * 6 * blockDim.x] = in[ti + 6 * blockDim.x];
out[to + ny * 7 * blockDim.x] = in[ti + 7 * blockDim.x];
}
}