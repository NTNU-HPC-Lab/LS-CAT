#include "includes.h"
__global__ void unsafe(int *shared_var, int *values_read, int N, int iters)
{
int i;
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid >= N) return;

int old = *shared_var;
*shared_var = old + 1;
values_read[tid] = old;

for (i = 0; i < iters; i++)
{
int old = *shared_var;
*shared_var = old + 1;
}
}