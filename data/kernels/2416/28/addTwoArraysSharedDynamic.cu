#include "includes.h"
__global__ void addTwoArraysSharedDynamic(int *v1, int *v2, int *r, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid >= n)
{
return;
}

extern __shared__ int arrays[];
int *s_v1 = arrays;
int *s_v2 = &s_v1[n];
int *s_r = &s_v2[n];

s_v1[tid] = v1[tid];

s_v2[tid] = v2[tid];

s_r[tid] = s_v1[tid] + s_v2[tid];
r[tid] = s_r[tid];
}