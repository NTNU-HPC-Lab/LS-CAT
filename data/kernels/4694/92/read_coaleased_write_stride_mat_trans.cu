#include "includes.h"
__global__ void read_coaleased_write_stride_mat_trans(float* input, float* output, const int nx, const int ny)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

if (ix < nx && iy < ny)
{
output[ix*ny + iy] = input[iy*nx + ix];
}
}