#include "includes.h"
__global__ void reduction(float *g_data, int n)
{
__shared__ float s_data[NUM_ELEMENTS];

int tid = threadIdx.x;

int myIndex = threadIdx.x + blockIdx.x*blockDim.x;

//s_data[tid] = 0.0;

s_data[tid] = g_data[myIndex];

__syncthreads();

for(int s = blockDim.x / 2; s > 0; s >>=1)
{
if(tid < s)
{

s_data[tid] += s_data[tid + s];

}

__syncthreads();
}

if (tid == 0)
{

g_data[blockIdx.x] = s_data[tid];

}


}