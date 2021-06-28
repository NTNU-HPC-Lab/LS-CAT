#include "includes.h"
__global__ void callOperationSharedDynamic(int * a, int *b, int *res, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n)
{
return;
}
int tid = tidx * n + tidy;

extern __shared__ int data[];

int *s_a = data;
int *s_b = &s_a[size * size];
int *s_res = &s_b[size * size];

s_a[tid] = a[tid];
s_b[tid] = b[tid];

s_res[tid] = s_a[tid] - s_b[tid];
if (s_res[tid] < 0)
{
s_res[tid] = 0;
}
res[tid] = s_res[tid];
}