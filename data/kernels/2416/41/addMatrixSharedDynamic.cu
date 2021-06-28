#include "includes.h"
__global__ void addMatrixSharedDynamic(int *a, int *b, int *res, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n)
{
return;
}

int tid = tidx * n + tidy;

extern __shared__ int arrays[];

int *s_a = arrays;
int *s_b = &arrays[size * size];
int *s_res = &s_b[size*size];

s_a[tid] = a[tid];
s_b[tid] = b[tid];

s_res[tid] = s_a[tid] + s_b[tid];
res[tid] = s_res[tid];
}