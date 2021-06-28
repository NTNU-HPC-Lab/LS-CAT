#include "includes.h"
__global__ void totalWithThreadSyncAndSharedMemInterleaved(float *input, float *output, int len) {
//@@ Compute reduction for a segment of the input vector
__shared__ float sdata[BLOCK_SIZE];
int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;

if(i  < len)
sdata[tid] = input[i];
else
sdata[tid] = 0.0;

for(unsigned int j = 1; j < blockDim.x; j *= 2)
{
if (tid % (2 * j) == 0)
sdata[tid] += sdata[tid+j];
__syncthreads();
}

if(tid == 0)
{
output[blockIdx.x] = sdata[0];
}
}