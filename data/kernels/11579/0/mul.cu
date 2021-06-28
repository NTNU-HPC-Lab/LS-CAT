#include "includes.h"
#define _size 512

__global__ void mul(int *a, int *b, int *c)
{
c[threadIdx.x + blockIdx.x*blockDim.x] = a[threadIdx.x + blockIdx.x*blockDim.x]*b[threadIdx.x + blockIdx.x*blockDim.x];
}