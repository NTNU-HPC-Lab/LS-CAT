#include "includes.h"
__global__ void addMatrix(int *a, int *b, int *res, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n) {
return;
}

int tid = tidx * n + tidy;
res[tid] = a[tid] + b[tid];
}