#include "includes.h"
__global__ void reduction_unrolling_blocks4(int * input, int * temp, int size)
{
int tid = threadIdx.x;

int BLOCK_OFFSET = blockIdx.x * blockDim.x * 4;

int index = BLOCK_OFFSET + tid;

int * i_data = input + BLOCK_OFFSET;

if ((index + 3 * blockDim.x) < size)
{
int a1 = input[index];
int a2 = input[index + blockDim.x];
int a3 = input[index+ 2* blockDim.x];
int a4 = input[index+ 3 *blockDim.x];
input[index] = a1 + a2 + a3 + a4;
}

__syncthreads();

for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
{
if (tid < offset)
{
i_data[tid] += i_data[tid + offset];
}

__syncthreads();
}

if (tid == 0)
{
temp[blockIdx.x] = i_data[0];
}
}