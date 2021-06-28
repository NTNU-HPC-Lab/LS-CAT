#include "includes.h"
__global__ void float2toUchar1(float2 *inputImage, uchar1 *outputImage, int width, int height, int index) {
int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
float2 pixelf = inputImage[offset];
float pixelfIndexed = (index == 0) ? pixelf.x : pixelf.y;
uchar1 pixel;
pixel.x = (unsigned char) pixelfIndexed;
outputImage[offset] = pixel;
}