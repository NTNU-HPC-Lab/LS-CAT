#include "includes.h"
__global__ void callOperationSharedStatic(int *a, int *b, int *c, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid >= n) {
return;
}

__shared__ int s_a[size], s_b[size], s_c[size];

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