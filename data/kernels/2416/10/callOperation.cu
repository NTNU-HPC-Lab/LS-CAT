#include "includes.h"
__global__ void callOperation(int *a, int *result, int k, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n)
{
return;
}

int tid = tidx * n + tidy;

if (a[tid] == k)
{
atomicAdd(result, 1);
}
}