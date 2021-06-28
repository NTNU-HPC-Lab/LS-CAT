#include "includes.h"
__global__ void copy_columns(float* input, float* output, const int nx, const int ny)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

if (ix < nx && iy < ny)
{
output[ix*ny + iy] = input[ix*ny + iy];
}
}