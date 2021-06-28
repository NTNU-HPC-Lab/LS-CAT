#include "includes.h"
__global__ void RGBToBGRA8(float3* srcImage, uchar4* dstImage, int width, int height, float scaling_factor)
{
const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

const int pixel = y * width + x;

if( x >= width )
return;

if( y >= height )
return;

const float3 px = srcImage[pixel];
dstImage[pixel] = make_uchar4(px.z * scaling_factor,
px.y * scaling_factor,
px.x * scaling_factor,
255.0f * scaling_factor);
}