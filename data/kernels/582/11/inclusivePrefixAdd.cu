#include "includes.h"
__global__ void inclusivePrefixAdd(unsigned int* d_in, unsigned int* d_out)
{
//Hillis Steele implementation
//NOTE: right now, this is only set up for 1 block of 1024 threads

int abs_x = threadIdx.x + blockIdx.x * blockDim.x;
int thread_x = threadIdx.x;

extern __shared__ unsigned int segment[];
segment[thread_x] = d_in[abs_x];
//d_out[thread_x] = d_in[thread_x];
__syncthreads();

for (unsigned int i = 1; i < blockDim.x; i <<= 1)
{
if (thread_x >= i)
{
//d_out[thread_x] = d_out[thread_x] + d_out[thread_x - i];
segment[thread_x] = segment[thread_x] + segment[thread_x - i];
}

__syncthreads();
}

//this happens in different blocks, so no need to syncthreads()
if (blockIdx.x > 0)
{
//carry over the result of the last segment
segment[thread_x] = segment[thread_x] + d_out[blockDim.x * (blockIdx.x - 1)];
}

d_out[abs_x] = segment[thread_x];
}