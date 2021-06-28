#include "includes.h"
__global__ void transposeUnroll4Col(int *in, int *out, const int nx, const int ny)
{
// set thread id.
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x * 4;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

unsigned int ti = iy * nx + ix; // access in rows.
unsigned int to = ix * ny + iy; // access in cols.

if (ix + 3 * blockDim.x < nx && iy < ny)
{
out[ti]                  = in[to];
out[ti + blockDim.x]     = in[to + ny * blockDim.x];
out[ti + blockDim.x * 2] = in[to + ny * blockDim.x * 2];
out[ti + blockDim.x * 3] = in[to + ny * blockDim.x * 3];
}
}