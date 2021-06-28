#include "includes.h"
__global__ void reduction_interleaved_unrolling_blocks2_1(int * input, int * temp, int size)
{
int tid = threadIdx.x;

//start index for this thread
int index = blockDim.x * blockIdx.x * 2 + threadIdx.x;

//local index for this block
int * i_data = input + blockDim.x * blockIdx.x * 2;

//unrolling two blocks
if ((index + blockDim.x)< size)
{
input[index] += input[index + blockDim.x];
}

__syncthreads();

for (int offset = blockDim.x / 2; offset > 0;
offset = offset / 2)
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