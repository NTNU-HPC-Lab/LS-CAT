#include "includes.h"



__global__ void TransposeKernelFast2(const uint8_t *src, uint8_t *dst, int width, int height)
{
int tx = threadIdx.x;
int ty = threadIdx.y;
int xbase = blockIdx.x * 32;
int ybase = blockIdx.y * 32;

__shared__ uint8_t sbuf[32][32+4];

{
int x = xbase + tx;
if (x < width) {
int yend = min(ybase + 32, height);
for (int tyy = ty, y = ybase + ty; y < yend; tyy += 8, y += 8) {
sbuf[tyy][tx] = src[x + y * width];
}
}
}

__syncthreads();

{
int y = ybase + tx;
if (y < height) {
int xend = min(xbase + 32, width);
for (int tyy = ty, x = xbase + ty; x < xend; tyy += 8, x += 8) {
dst[y + x * height] = sbuf[tx][tyy];
}
}
}
}