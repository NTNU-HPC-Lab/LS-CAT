#include "includes.h"
__global__ void stencilReadOnly3(float *src, float *dst, int size, float* stencilWeight)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
idx += 11;
if (idx >= size)
return;
float out = 0;
#pragma unroll
for(int i = -10;i < 10; i++)
{
out += src[idx+i] * stencilWeight[i+10];
}
dst[idx] = out;
}