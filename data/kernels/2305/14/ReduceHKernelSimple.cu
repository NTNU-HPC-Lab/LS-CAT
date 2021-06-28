#include "includes.h"



__global__ void ReduceHKernelSimple(const uint8_t *src, float *dst, int width, int height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;

if (x < width) {
float sum = 0;
for (int y = 0; y < height; ++y) {
sum += src[x + y * width];
}
dst[x] = sum;
}
}