#include "includes.h"
__global__ void callOperation(int * a, int *b, int *res, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;

if (tid >= n)
{
return;
}

res[tid] = a[tid] - b[tid];
if (res[tid] < 0)
{
res[tid] = 0;
}
}