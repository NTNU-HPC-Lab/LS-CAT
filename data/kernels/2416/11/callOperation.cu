#include "includes.h"
__global__ void callOperation(int *a, int *res, int x, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n) {
return;
}

int tid = tidx * n + tidy;

res[tid] = a[tid] * x;
}