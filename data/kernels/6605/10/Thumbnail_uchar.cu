#include "includes.h"
__global__ void Thumbnail_uchar(cudaTextureObject_t uchar_tex, int *histogram, int src_width, int src_height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if (y < src_height && x < src_width)
{
unsigned char pixel = tex2D<unsigned char>(uchar_tex, x, y);
atomicAdd(&histogram[pixel], 1);
}
}