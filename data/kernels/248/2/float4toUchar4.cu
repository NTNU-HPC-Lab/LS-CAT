#include "includes.h"
__global__ void float4toUchar4(float4 *inputImage, uchar4 *outputImage, int width, int height)
{
int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

float4 pixelf = inputImage[offset];
uchar4 pixel;
pixel.x = (unsigned char) pixelf.x; pixel.y = (unsigned char) pixelf.y;
pixel.z = (unsigned char) pixelf.z; pixel.w = (unsigned char) pixelf.w;

outputImage[offset] = pixel;
}