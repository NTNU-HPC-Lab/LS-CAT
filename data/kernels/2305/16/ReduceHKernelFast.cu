#include "includes.h"



__global__ void ReduceHKernelFast(const uint8_t *src, float *dst, int width, int height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * 128;

if (x < width) {
float sum = 0;
for (int yend = min(y + 128, height); y < yend; ++y) {
sum += src[x + y * width];
}
atomicAdd(&dst[x], sum);
}
}