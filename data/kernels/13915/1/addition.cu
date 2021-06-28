#include "includes.h"
__global__ void addition(int *a, int *b, int *c)
{
int tid = blockIdx.x;
if (tid < N)
c[tid] = a[tid] + b[tid];
}