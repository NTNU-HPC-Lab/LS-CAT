#include "includes.h"
__global__ void Thumbnail_uchar2(cudaTextureObject_t uchar2_tex, int *histogram, int src_width, int src_height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (y < src_height && x < src_width)
{
uchar2 pixel = tex2D<uchar2>(uchar2_tex, x, y);
atomicAdd(&histogram[pixel.x], 1);
atomicAdd(&histogram[256 + pixel.y], 1);
}
}