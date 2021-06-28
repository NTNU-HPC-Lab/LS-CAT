#include "includes.h"



__global__ void TransposeKernelShared(const uint8_t *src, uint8_t *dst, int width, int height)
{
int tx = threadIdx.x;
int ty = threadIdx.y;
int xbase = blockIdx.x * blockDim.x;
int ybase = blockIdx.y * blockDim.y;

__shared__ uint8_t sbuf[16][16];

{
int x = xbase + tx;
int y = ybase + ty;
if (x < width && y < height)
sbuf[ty][tx] = src[x + y * width];
}

__syncthreads();

{
int x = xbase + ty;
int y = ybase + tx;
if (x < width && y < height)
dst[y + x * height] = sbuf[tx][ty];
}
}