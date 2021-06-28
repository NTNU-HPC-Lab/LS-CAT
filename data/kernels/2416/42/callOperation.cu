#include "includes.h"
__global__ void callOperation(int *a, int *b, int *res, int k, int p, int n)
{
int idx = blockDim.x * blockIdx.x + threadIdx.x;
int idy = blockDim.y * blockIdx.y + threadIdx.y;

if (idx >= n || idy >= n) {
return;
}

int tid = idx * n + idy;

res[tid] = a[tid] + b[tid];

if (res[tid] > k) {
res[tid] = p;
}
}