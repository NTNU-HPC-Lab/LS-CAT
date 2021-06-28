#include "includes.h"



__global__ void ReduceWKernelSimple(const uint8_t *src, float *dst, int width, int height)
{
int y = blockIdx.x * blockDim.x + threadIdx.x;
int x = blockIdx.y * 128;

if (y < height) {
float sum = 0;
for (int xend = min(x + 128, width); x < xend; ++x) {
sum += src[x + y * width];
}
atomicAdd(&dst[y], sum);
}
}