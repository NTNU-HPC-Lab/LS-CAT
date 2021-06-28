#include "includes.h"
__global__ void stencilShared1(float *src, float *dst, int size, int raio)
{
int idx = blockIdx.x * blockDim.x + threadIdx.x;
__shared__ float buffer[1024+11];
for(int i = threadIdx.x; i < 1024+21; i = i + 1024)
{
buffer[i] = src[idx+i];
}
idx += raio+1;
if (idx >= size)
return;

__syncthreads();
float out = 0;
#pragma unroll
for(int i = -raio;i < raio; i++)
{
out += buffer[threadIdx.x+raio+i] * const_stencilWeight[i+raio];
}
dst[idx] = out;
}