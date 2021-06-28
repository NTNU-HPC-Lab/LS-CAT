#include "includes.h"
__global__ void convertRGBtoGrayScale(uint8_t* src, uint8_t* dst,int width,int height, int channels)
{
int x = threadIdx.x+ blockIdx.x* blockDim.x;
int y = threadIdx.y+ blockIdx.y* blockDim.y;
if(x < width && y < height) {
int grayOffset= y*width + x;// one can think of the RGB image having
int rgbOffset= grayOffset*channels;// CHANNEL times columns than the gray scale
unsigned char r =  src[rgbOffset]; // red value for pixel
unsigned char g = src[rgbOffset+ 2]; // green value for pixel
unsigned char b = src[rgbOffset+ 3]; // blue value for pixel// perform the rescaling and store it// We multiply by floating point constants
dst[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
}
}