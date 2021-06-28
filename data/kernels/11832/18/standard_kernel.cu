#include "includes.h"
__global__ void standard_kernel(float a, float *out, int iters)
{
int i;
int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

if(tid == 0)
{
float tmp;

for (i = 0; i < iters; i++)
{
tmp = powf(a, 2.0f);
}

*out = tmp;
}
}