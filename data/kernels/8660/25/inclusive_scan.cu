#include "includes.h"
__global__ void inclusive_scan(const unsigned int *input, unsigned int *result)
{
extern __shared__ unsigned int sdata[];

unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

// load input into __shared__ memory
unsigned int sum = input[i];
sdata[threadIdx.x] = sum;
__syncthreads();
for(int offset = 1; offset < blockDim.x; offset <<= 1)
{
if(threadIdx.x >= offset)
{
sum += sdata[threadIdx.x - offset];
}

// wait until every thread has updated its partial sum
__syncthreads();

// write my partial sum
sdata[threadIdx.x] = sum;

// wait until every thread has written its partial sum
__syncthreads();
}

// we're done! each thread writes out its result
result[i] = sdata[threadIdx.x];
}