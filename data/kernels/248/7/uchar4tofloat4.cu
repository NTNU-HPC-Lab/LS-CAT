#include "includes.h"
__global__ void uchar4tofloat4(uchar4 *inputImage, float4 *outputImage, int width, int height)
{
int offsetX = blockIdx.x * blockDim.x + threadIdx.x;
int offsetY = blockIdx.y * blockDim.y + threadIdx.y;

if (offsetX < width && offsetY < height)
{
int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

uchar4 pixel = inputImage[offset];
float4 pixelf;
pixelf.x = pixel.x; pixelf.y = pixel.y;
pixelf.z = pixel.z; pixelf.w = pixel.w;

outputImage[offset] = pixelf;
}
}