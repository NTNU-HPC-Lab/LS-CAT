#include "includes.h"
__global__ void sum_arrays_2Dgrid_2Dblock(float* a, float* b, float *c, int nx, int ny)
{
int gidx = blockIdx.x * blockDim.x + threadIdx.x;
int gidy = blockIdx.y*blockDim.y + threadIdx.y;

int gid = gidy * nx + gidx;

if(gidx < nx && gidy < ny)
c[gid] = a[gid] + b[gid];
}