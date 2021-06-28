#include "includes.h"

__global__ void transposeGlobalRow(float *in, float *out, const int nx, const int ny)
{
unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
unsigned int j = threadIdx.y+blockDim.y*blockIdx.y;

if (i<nx && j<ny)
{
out[i*ny+j] = in[j*nx+i];
}
}