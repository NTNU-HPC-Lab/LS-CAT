#include "includes.h"

__global__ void convertBGR2RGBfloatKernel(uchar3 *src, float3 *dst, int width, int height)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
if (x >= width || y >= height) {
return;
}

uchar3 color = src[y * width + x];
dst[y * width + x] = make_float3(color.z, color.y, color.x);
}