#include "includes.h"
__global__ void redunction_neighbored_pairs_1(int * input, int * temp, int size)
{
int tid = threadIdx.x;
int gid = blockDim.x * blockIdx.x + threadIdx.x;

if (gid > size)
return;

for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
{
if (tid % (2 * offset) == 0)
{
input[gid] += input[gid + offset];
}

__syncthreads();
}

if (tid == 0)
{
temp[blockIdx.x] = input[gid];
}
}