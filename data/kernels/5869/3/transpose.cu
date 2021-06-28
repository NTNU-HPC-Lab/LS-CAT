#include "includes.h"
__global__ void transpose(double *in_d, double * out_d, int row, int col)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
out_d[y+col*x] = in_d[x+row*y];
}