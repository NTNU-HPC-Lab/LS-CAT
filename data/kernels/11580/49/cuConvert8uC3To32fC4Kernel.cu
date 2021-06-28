#include "includes.h"
__global__ void cuConvert8uC3To32fC4Kernel(const unsigned char *src, size_t src_pitch, float4* dst, size_t dst_stride, float mul_constant, float add_constant, int width, int height)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int src_c = y*src_pitch + x*3;
int dst_c = y*dst_stride + x;

if (x<width && y<height)
{
dst[dst_c] = make_float4(src[src_c]/255.0f, src[src_c+1]/255.0f, src[src_c+2]/255.0f, 1.0f);// * mul_constant + add_constant;
}
}