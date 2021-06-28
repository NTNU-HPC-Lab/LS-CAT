#include "includes.h"

__global__ void imageSplitKernel(float3 *ptr, float *dst, int width, int height)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
if (x >= width || y >= height) {
return;
}

float3 color = ptr[y * width + x];

dst[y * width + x] = color.x;
dst[y * width + x + width * height] = color.y;
dst[y * width + x + width * height * 2] = color.z;
}