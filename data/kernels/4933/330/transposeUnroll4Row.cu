#include "includes.h"
__global__ void transposeUnroll4Row(int *in, int *out, const int nx, const int ny)
{
// set thread id.
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x * 4;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

unsigned int ti = iy * nx + ix; // access in rows.
unsigned int to = ix * ny + iy; // access in cols.

if (ix + 3 * blockDim.x < nx && iy < ny)
{
out[to]                       = in[ti];
out[to + ny * blockDim.x]     = in[ti + blockDim.x];
out[to + ny * blockDim.x * 2] = in[ti + blockDim.x * 2];
out[to + ny * blockDim.x * 3] = in[ti + blockDim.x * 3];
}
}