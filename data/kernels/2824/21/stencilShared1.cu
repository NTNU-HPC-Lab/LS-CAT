#include "includes.h"
__global__ void stencilShared1(float *src, float *dst, int size)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
__shared__ float buffer[1024+21];
for(int i = threadIdx.x; i < 1024+21; i = i + 1024)
{
buffer[i] = src[idx+i];
}
idx += 11;
if (idx >= size)
return;

__syncthreads();
float out = 0;
#pragma unroll
for(int i = -10;i < 10; i++)
{
out += buffer[threadIdx.x+10+i] * const_stencilWeight[i+10];
}
dst[idx] = out;
}