#include "includes.h"

__global__ void imageNormalizationKernel(float3 *ptr, int width, int height)
{
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
if (x >= width || y >= height) {
return;
}

float3 color = ptr[y * width + x];
color.x = (color.x - 127.5) * 0.0078125;
color.y = (color.y - 127.5) * 0.0078125;
color.z = (color.z - 127.5) * 0.0078125;

ptr[y * width + x] = make_float3(color.x, color.y, color.z);
}