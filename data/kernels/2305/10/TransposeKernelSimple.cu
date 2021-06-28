#include "includes.h"



__global__ void TransposeKernelSimple(const uint8_t *src, uint8_t *dst, int width, int height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < width && y < height)
dst[y + x * height] = src[x + y * width];
}