#include "includes.h"
__global__ void addTwoArrays(int *v1, int *v2, int *r, int n)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid >= n) {
return;
}

r[tid] = v1[tid] + v2[tid];
}