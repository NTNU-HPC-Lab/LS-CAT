#include "includes.h"
__global__ void RGBToRGBAf(uchar3* srcImage, float4* dstImage, int width, int height)
{
const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

const int pixel = y * width + x;

if( x >= width )
return;

if( y >= height )
return;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);

const float  s  = 1.0f;
const uchar3 px = srcImage[pixel];

dstImage[pixel] = make_float4(px.x * s, px.y * s, px.z * s, 255.0f * s);
}