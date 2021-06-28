#include "includes.h"
__global__ void cuConvertC3ToC4Kernel(const float3* src, size_t src_stride, float4* dst, size_t dst_stride, int width, int height)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int c_src = y*src_stride + x;
int c_dst = y*dst_stride + x;

if (x<width && y<height)
{
float3 val=src[c_src];
dst[c_dst] =  make_float4(val.x, val.y, val.z, 1.0f);
}
}