#include "includes.h"
__global__ void kernel( int *a, int dimx, int dimy )
{
int ix   = blockIdx.x*blockDim.x + threadIdx.x;
int iy   = blockIdx.y*blockDim.y + threadIdx.y;
int idx = iy*dimx + ix;

a[idx]  = a[idx]+1;
}