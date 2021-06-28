#include "includes.h"
__global__ void callOperationSharedDynamic(int *a, int *b, int *res, int x, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid >= n)
{
return;
}

extern __shared__ int arrays[];
__shared__ int s_x;

int *s_a = arrays;
int *s_b = &s_a[n];
int *s_res = &s_b[n];

s_x = x;
s_a[tid] = a[tid];
s_b[tid] = b[tid];

s_res[tid] = s_a[tid] - (s_b[tid] * s_x);
res[tid] = s_res[tid];
}