#include "includes.h"
__global__ void totalWithThreadSync(float *input, float *output, int len) {
//@@ Compute reduction for a segment of the input vector
int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;

for(unsigned int j = blockDim.x/2; j > 0; j = j/2)
{
if(tid < j)
{
if ((i + j) < len)
input[i] += input[i+j];
else
input [i] += 0.0;
}
__syncthreads();
}

if(tid == 0)
{
output[blockIdx.x] = input[i];
}
}