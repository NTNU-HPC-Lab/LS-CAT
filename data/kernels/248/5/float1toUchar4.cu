#include "includes.h"
__global__ void float1toUchar4(float1 *inputImage, uchar4 *outputImage, int width, int height)
{
int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

float1 pixelf = inputImage[offset];
uchar4 pixel;
pixel.x = (unsigned char) pixelf.x; pixel.y = (unsigned char) pixelf.x;
pixel.z = (unsigned char) pixelf.x; pixel.w = (unsigned char) pixelf.x;

outputImage[offset] = pixel;
}