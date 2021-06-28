#include "includes.h"
__global__ void atomics(int *shared_var, int *values_read, int N, int iters)
{
int i;
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if (tid >= N) return;

values_read[tid] = atomicAdd(shared_var, 1);

for (i = 0; i < iters; i++)
{
atomicAdd(shared_var, 1);
}
}