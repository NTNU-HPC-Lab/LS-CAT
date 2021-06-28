#include "includes.h"
__global__ void totalSequentialSharedMem(float *input, float *output, int len) {
//@@ Compute reduction for a segment of the input vector
int tid = threadIdx.x, i = blockIdx.x * blockDim.x;
__shared__ float sdata[BLOCK_SIZE];
sdata[tid] = i + tid < len ? input[i+tid] : 0.0;

if(tid == 0) {
for(unsigned int j = 1; j <blockDim.x; j++)
{
sdata[0] += sdata[j];
}
output[blockIdx.x] = sdata[0];
}
}