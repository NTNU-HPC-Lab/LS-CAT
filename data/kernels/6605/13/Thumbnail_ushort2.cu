#include "includes.h"
__global__ void Thumbnail_ushort2(cudaTextureObject_t ushort2_tex, int *histogram, int src_width, int src_height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (y < src_height && x < src_width)
{
ushort2 pixel = tex2D<ushort2>(ushort2_tex, x, y);
atomicAdd(&histogram[(pixel.x + 128) >> 8], 1);
atomicAdd(&histogram[256 + (pixel.y + 128) >> 8], 1);
}
}