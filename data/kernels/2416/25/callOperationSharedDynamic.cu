#include "includes.h"
__global__ void callOperationSharedDynamic(int *a, int *b, int *c, int n)
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
int *s_c = &s_b[size * size];

s_a[tid] = a[tid];
s_b[tid] = b[tid];

if (s_a[tid] >= s_b[tid])
{
s_c[tid] = s_a[tid];
}
else
{
s_c[tid] = s_b[tid];
}

c[tid] = s_c[tid];
}