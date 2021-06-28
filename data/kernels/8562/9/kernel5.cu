#include "includes.h"
__global__ void kernel5( int *a, int dimx, int dimy )
{
int ix = blockIdx.x * blockDim.x + threadIdx.x;
int iy = blockIdx.y * blockDim.y + threadIdx.y;
int idx = iy * dimx + ix;
if(ix<dimx && iy < dimy)
a[idx] = blockIdx.y;
}