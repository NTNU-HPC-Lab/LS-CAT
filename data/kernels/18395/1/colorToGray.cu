#include "includes.h"
__global__ void colorToGray(unsigned char *input, unsigned char *output, int height, int width)
{
int col= blockDim.x * blockIdx.x + threadIdx.x;
int row = blockDim.y * blockIdx.y + threadIdx.y;
float scale[3] = {0.299, 0.587, 0.114};
if (row < height && col < width)
{
int pixelIndex = row * width + col;
int rgbIndex = pixelIndex * 3;

unsigned char r = input[rgbIndex];                 // rgb rgb rgb rgb rgb
unsigned char g = input[rgbIndex + 1];
unsigned char b = input[rgbIndex + 2];
output[pixelIndex] = r* scale[0] + g * scale[1] + b*scale[2];
}
}