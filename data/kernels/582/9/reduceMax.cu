#include "includes.h"
__global__ void reduceMax(const float* d_in, float* d_out)
{
int abs_x = threadIdx.x + blockIdx.x * blockDim.x;
int thread_x = threadIdx.x;

extern __shared__ float sdata[];

sdata[thread_x] = d_in[abs_x];
__syncthreads();

int last_i = blockDim.x;
for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1)
{
if (thread_x < i)
{
sdata[thread_x] = max(sdata[thread_x], sdata[thread_x + i]);

//this checks for weird edge cases where the block dimension is not a power of two
//see https://discussions.udacity.com/t/wrong-max-value-problem-set-3/85232/7

//basically, if we are at the "last" thread of this iteration (i - 1)
//and if we lost a point due to integer divison
if (thread_x == i - 1 && last_i > 2 * i)
{
//then take the point we lost to integer divison at (last_i - 1)
sdata[thread_x] = max(sdata[thread_x], sdata[last_i - 1]);
}
}

__syncthreads();
last_i = i;
}

//return result at the 0th thread of every block:
if (thread_x == 0)
{
d_out[blockIdx.x] = sdata[0];
}
}