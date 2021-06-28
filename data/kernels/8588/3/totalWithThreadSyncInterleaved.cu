#include "includes.h"
__global__ void totalWithThreadSyncInterleaved(float *input, float *output, int len) {
//@@ Compute reduction for a segment of the input vector
int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;

for(unsigned int j = 1; j <blockDim.x; j *= 2)
{
if (tid % (2 * j) == 0)
input[i] += input[i+j];
__syncthreads();
}

if(tid == 0)
{
output[blockIdx.x] = input[i];
}
}