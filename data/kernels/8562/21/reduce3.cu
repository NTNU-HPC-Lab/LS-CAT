#include "includes.h"
__global__ void reduce3(float *in, float *out, int size)
{
extern __shared__ float sdata[];
int tid = threadIdx.x;
int index = blockIdx.x * blockDim.x + threadIdx.x;
sdata[tid] = (index < size) ? in[index] : 0;

__syncthreads();
for(int s = blockDim.x/2; s>0; s>>=1)
{
if(tid<s)
sdata[tid] += sdata[tid + s];
__syncthreads();
}//end of for loop
if(tid == 0)
out[blockIdx.x] = sdata[0];
}//end of reduce3 kernal