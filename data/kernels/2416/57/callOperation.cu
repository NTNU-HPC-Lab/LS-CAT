#include "includes.h"
__global__ void callOperation(int *a, int *b, int *c, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid >= n) {
return;
}

if (a[tid] <= b[tid])
{
c[tid] = a[tid];
}
else
{
c[tid] = b[tid];
}
}