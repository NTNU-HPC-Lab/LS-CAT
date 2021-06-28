#include "includes.h"
__global__ void float1toUchar1(float1 *inputImage, uchar1 *outputImage, int width, int height)
{
int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
int offset = offsetBlock + threadIdx.x + threadIdx.y * width;

float1 pixelf = inputImage[offset];
uchar1 pixel;
pixel.x = (unsigned char) pixelf.x;

outputImage[offset] = pixel;
}