#include "includes.h"
__global__ void calculateMatrixFormulaSharedStatic(int *a, int *b, int *res, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n) {
return;
}

int tid = tidx * n + tidy;

__shared__ int s_a[size * size], s_b[size * size], s_res[size * size];

s_a[tid] = a[tid];
s_b[tid] = b[tid];

s_res[tid] = s_a[tid] - s_b[tid];
res[tid] = s_res[tid];
}