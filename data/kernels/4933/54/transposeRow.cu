#include "includes.h"
__global__ void transposeRow(float *out, float *in, const int nx, const int ny)
{
unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
unsigned int row = iy * gridDim.x * blockDim.x + ix;

if (row < ny)
{
int row_start = row * nx;
int row_end = (row + 1) * nx;
int col_index = row;
for (int i = row_start; i < row_end; i++) {
out[col_index] = in[i];
col_index += nx;
}
}
}