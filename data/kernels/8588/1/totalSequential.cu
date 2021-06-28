#include "includes.h"
__global__ void totalSequential(float *input, float *output, int len) {
//@@ Compute reduction for a segment of the input vector
int tid = threadIdx.x, i = blockIdx.x * blockDim.x;

if(tid == 0) {
int sum = 0;
for(unsigned int j = 0; j <blockDim.x; j++)
{
sum += input[i + j];
}
output[blockIdx.x] = sum;
}
}