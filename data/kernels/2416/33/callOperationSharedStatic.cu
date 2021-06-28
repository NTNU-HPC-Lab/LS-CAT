#include "includes.h"
__global__ void callOperationSharedStatic(int *a, int *b, int *c, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n)
{
return;
}

int tid = tidx * n + tidy;

__shared__ int s_a[size * size], s_b[size * size], s_c[size * size];

s_a[tid] = a[tid];
s_b[tid] = b[tid];

if (s_a[tid] <= s_b[tid])
{
s_c[tid] = s_a[tid];
}
else
{
s_c[tid] = s_b[tid];
}

c[tid] = s_c[tid];
}