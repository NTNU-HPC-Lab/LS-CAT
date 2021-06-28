#include "includes.h"
__global__ void totalWithThreadSyncAndSharedMem(float *input, float *output, int len) {
//@@ Compute reduction for a segment of the input vector
__shared__ float sdata[BLOCK_SIZE];
int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;

if(tid < len)
sdata[tid] = input[i];
else
sdata[tid] = 0.0;

__syncthreads();

for(unsigned int j = blockDim.x/2; j > 0; j = j/2)
{
if(tid < j)
{
sdata[tid] += sdata[tid+j];
}
__syncthreads();
}

if(tid == 0)
{
output[blockIdx.x] = sdata[0];
}
}