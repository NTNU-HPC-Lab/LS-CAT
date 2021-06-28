#include "includes.h"
__global__ void copy_rows(float* input, float* output, const int nx, const int ny)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

if (ix < nx && iy < ny)
{
output[iy*ny + ix] = input[iy*nx + ix];
}
}