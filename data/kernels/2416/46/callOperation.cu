#include "includes.h"
__global__ void callOperation(int *a, int *b, int x, int *res, int n)
{
int tid = blockDim.x * blockIdx.x + threadIdx.x;
if (tid < n) {
res[tid] = ((a[tid] * x) + b[tid]);
}
}