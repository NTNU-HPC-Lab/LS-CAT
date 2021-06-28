#include "includes.h"
__global__ void exclusivePrefixAdd(unsigned int* d_in, unsigned int* d_out)
{
//Belloch implementation
//NOTE: this is set up specifically for 1 block of 1024 threads

int thread_x = threadIdx.x;

d_out[thread_x] = d_in[thread_x];
__syncthreads();

//first, do the reduce:
for (unsigned int i = 2; i <= blockDim.x; i <<= 1)
{
if ((thread_x + 1) % i == 0)
{
d_out[thread_x] = d_out[thread_x] + d_out[thread_x - i / 2];
}

__syncthreads();
}


//now do the downsweep part:

if (thread_x == blockDim.x - 1)
{
d_out[thread_x] = 0;
}

//maybe need a syncthreads() here because of that write above? it's only 1 thread so idk if it affects it

for (unsigned int i = blockDim.x; i >= 2; i >>= 1)
{
if ((thread_x + 1) % i == 0)
{
unsigned int temp = d_out[thread_x - (i / 2)];

//the "left" copy
d_out[thread_x - (i / 2)] = d_out[thread_x];

//and the "right" operation
d_out[thread_x] = temp + d_out[thread_x];
}
__syncthreads();
}

}