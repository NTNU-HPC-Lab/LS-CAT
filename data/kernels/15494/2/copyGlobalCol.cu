#include "includes.h"

__global__ void copyGlobalCol(float *out, float *in, const int nx, const int ny)
{
unsigned int i = threadIdx.x+blockDim.x*blockIdx.x;
unsigned int j = threadIdx.y+blockDim.y*blockIdx.y;

if (i<nx && j<ny)
{
out[i*ny+j] = in[i*ny+j];
}
}