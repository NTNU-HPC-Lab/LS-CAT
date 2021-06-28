#include "includes.h"
__global__ void read_stride_write_coaleased_mat_trans(float* input, float* output, const int nx, const int ny)
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;

if (ix < nx && iy < ny)
{
output[iy*nx + ix] = input[ix*ny + iy];
}
}