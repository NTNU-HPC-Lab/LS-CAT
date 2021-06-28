#include "includes.h"
__global__ void cuConvertC4ToC3Kernel(const float4* src, size_t src_stride, float3* dst, size_t dst_stride, int width, int height)
{
const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;
int src_c = y*src_stride + x;
int dst_c = y*dst_stride + x;

if (x<width && y<height)
{
float4 val=src[src_c];
dst[dst_c] = make_float3(val.x, val.y, val.z);
}
}