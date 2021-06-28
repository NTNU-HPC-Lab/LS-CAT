#include "includes.h"
__global__ void callOperationSharedDynamic(int *a, int *res, int x, int n)
{
int tidx = blockDim.x * blockIdx.x + threadIdx.x;
int tidy = blockDim.y * blockIdx.y + threadIdx.y;

if (tidx >= n || tidy >= n) {
return;
}

int tid = tidx * n + tidy;

extern __shared__ int data[];

int *s_a = data;
int *s_res = &s_a[size * size];

__shared__ int s_x;

s_x = x;
s_a[tid] = a[tid];

s_res[tid] = s_a[tid] * s_x;

res[tid] = s_res[tid];
}