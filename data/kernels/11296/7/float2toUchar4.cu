#include "includes.h"
__global__ void float2toUchar4(float2 *inputImage, uchar4 *outputImage, int width, int height, int index) {
int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
float2 pixelf = inputImage[offset];
float pixelfIndexed = (index == 0) ? pixelf.x : pixelf.y;
uchar4 pixel;
pixel.x = (unsigned char) abs(pixelfIndexed);
pixel.y = (unsigned char) abs(pixelfIndexed);
pixel.z = (unsigned char) abs(pixelfIndexed);
pixel.w = (unsigned char) abs(pixelfIndexed);
outputImage[offset] = pixel;
}