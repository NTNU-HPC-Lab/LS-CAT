#include "includes.h"
__global__ void find_max(int* input, int* result, int n)
{
__shared__ int sdata[size];
unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
unsigned int tx = threadIdx.x;
int x = -INT_MAX;

if (i<n)
{
x = input[i];
}
sdata[tx] = x;
__syncthreads();
for(unsigned int s = blockDim.x >> 1 ; s>0 ; s>>=1)
{
__syncthreads();
if(tx<s)
{
if(sdata[tx]>sdata[tx+s])
sdata[tx]=sdata[tx+s];
}
}
if (threadIdx.x == 0)
{
result[blockIdx.x] = sdata[0];
}
}