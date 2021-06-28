#include "includes.h"
__global__ void stencilConst1(float *src, float *dst, int size, int raio)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
idx += raio+1;
if (idx >= size)
return;
float out = 0;
#pragma unroll
for(int i = -raio;i < raio; i++)
{
out += src[idx+i] * const_stencilWeight[i+raio];
}
dst[idx] = out;
}