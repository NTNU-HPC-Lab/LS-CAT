#include "includes.h"
__global__ void reduction_neighbored_pairs_improved_1( int * int_array, int * temp_array, int size)
{
int tid = threadIdx.x;
int gid = blockDim.x * blockIdx.x + threadIdx.x;

//local data block pointer
int * i_data = int_array + blockDim.x * blockIdx.x;

if (gid > size)
return;

for (int offset = 1; offset <= blockDim.x / 2; offset *= 2)
{
int index = 2 * offset * tid;

if (index < blockDim.x)
{
i_data[index] += i_data[index + offset];
}

__syncthreads();
}

if (tid == 0)
{
temp_array[blockIdx.x] = int_array[gid];
}
}