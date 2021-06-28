#include "includes.h"
__global__ void MatMulInt(int *a, int b, int *c,int ROW, int COLUMNS){

int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;
int idx = iy * COLUMNS + ix;

if (ix < ROW && iy < COLUMNS)
{
c[idx] = a[idx] * b ;
}
}