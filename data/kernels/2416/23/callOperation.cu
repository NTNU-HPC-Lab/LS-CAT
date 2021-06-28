#include "includes.h"
__global__ void callOperation(int *a, int *b, int *c, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n)
{
return;
}

int tid = tidx * n + tidy;

if (a[tid] >= b[tid])
{
c[tid] = a[tid];
}
else
{
c[tid] = b[tid];
}
}