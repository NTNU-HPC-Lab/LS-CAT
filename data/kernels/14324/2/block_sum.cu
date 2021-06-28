#include "includes.h"
__global__ void block_sum(const int *input, int *per_block_results, const size_t n)
{
extern __shared__ int sdata[];

unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

// load input into __shared__ memory
int x = 0;
if(i < n)
{
x = input[i];
}
sdata[threadIdx.x] = x;
__syncthreads();

// contiguous range pattern
for(int offset = blockDim.x / 2;
offset > 0;
offset >>= 1)
{
if(threadIdx.x < offset)
{
// add a partial sum upstream to our own
sdata[threadIdx.x] += sdata[threadIdx.x + offset];
}

// wait until all threads in the block have
// updated their partial sums
__syncthreads();
}

// thread 0 writes the final result
if(threadIdx.x == 0)
{
per_block_results[blockIdx.x] = sdata[0];
}
}