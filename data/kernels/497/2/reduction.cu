#include "includes.h"
__global__ void reduction(float *g_data, int n)
{
__shared__ float partialSum[NUM_ELEMENTS];
unsigned int t = threadIdx.x;
partialSum[t] = g_data[t];

for (int i = blockDim.x/2; i > 0; i>>=1)
{
__syncthreads();
if(t<i)
{
partialSum[t] += partialSum[t + i];
}
}
if(t==0)
{
g_data[0] = partialSum[0];
}
}