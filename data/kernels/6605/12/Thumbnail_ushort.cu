#include "includes.h"
__global__ void Thumbnail_ushort(cudaTextureObject_t ushort_tex, int *histogram, int src_width, int src_height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (y < src_height && x < src_width)
{
unsigned short pixel = (tex2D<unsigned short>(ushort_tex, x, y) + 128) >> 8;
atomicAdd(&histogram[pixel], 1);
}
}