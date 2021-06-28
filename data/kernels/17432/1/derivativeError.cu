#include "includes.h"

#define BLOCK_SIZE 32


__global__ void derivativeError(float *output, float *actual, float *deriv_err)
{
__shared__ float sdata[1024];

//ideally block is 1024x1 and grid is ??? x units
int row = blockIdx.x * blockDim.x + threadIdx.x;
int col = blockIdx.y;

sdata[threadIdx.x] = output[row*gridDim.y + col];
__syncthreads();

for(int s= blockDim.x / 2; s>0; s>>=1)
{
if(threadIdx.x < s)
sdata[threadIdx.x] += sdata[threadIdx.x+s];
__syncthreads();
}
if(threadIdx.x == 0) //only tid0 can write
{
/*deriv_err[blockIdx.x] = sdata[0]*/deriv_err[blockDim.y*blockIdx.x+col] = sdata[blockIdx.x];
}
}