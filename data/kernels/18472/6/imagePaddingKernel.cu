#include "includes.h"

__global__ void imagePaddingKernel(float3 *ptr, float3 *dst, int width, int height, int top, int bottom, int left, int right)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
if(x < left || x >= (width - right) || y < top || y > (height - bottom)) {
return;
}

float3 color = ptr[(y - top) * (width - top - right) + (x - left)];

dst[y * width + x] = color;
}