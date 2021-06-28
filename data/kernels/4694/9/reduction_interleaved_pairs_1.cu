#include "includes.h"
__global__ void reduction_interleaved_pairs_1(int * int_array, int * temp_array, int size)
{
int tid = threadIdx.x;
int gid = blockDim.x * blockIdx.x + threadIdx.x;

if (gid > size)
return;

for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
{
if (tid < offset)
{
int_array[gid] += int_array[gid + offset];
}

__syncthreads();
}

if (tid == 0)
{
temp_array[blockIdx.x] = int_array[gid];
}
}