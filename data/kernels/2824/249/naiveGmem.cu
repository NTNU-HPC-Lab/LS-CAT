#include "includes.h"
__global__ void naiveGmem(float *out, float *in, const int nx, const int ny)
{
// matrix coordinate (ix,iy)
unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

// transpose with boundary test
if (ix < nx && iy < ny)
{
out[ix * ny + iy] = in[iy * nx + ix];
}
}