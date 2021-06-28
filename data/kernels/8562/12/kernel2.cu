#include "includes.h"
__global__ void kernel2( int *a, int dimx, int dimy )
{
int ix = blockIdx.x*blockDim.x + threadIdx.x;
int iy = blockIdx.y*blockDim.y + threadIdx.y;
int idx = iy*dimx + ix;
a[idx] = (blockIdx.x + blockIdx.y);
}