#include "includes.h"
__global__ void addTwoArraysSharedStatic(int *v1, int *v2, int *r, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid >= n)
{
return;
}

__shared__ int s_v1[SIZE], s_v2[SIZE], s_r[SIZE];

s_v1[tid] = v1[tid];

s_v2[tid] = v2[tid];

s_r[tid] = s_v1[tid] + s_v2[tid];
r[tid] = s_r[tid];
}