#include "includes.h"
__global__ void cuConvert32uC1To32fC1Kernel(const unsigned int *src, size_t src_stride, float* dst, size_t dst_stride, float mul_constant, float add_constant, int width, int height)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int src_c = y*src_stride + x;
int dst_c = y*dst_stride + x;

if (x<width && y<height)
{
dst[dst_c] = src[src_c] * mul_constant + add_constant;
}
}