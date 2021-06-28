#include "includes.h"
__global__ void bgr_to_gray_kernel(unsigned char* input, unsigned char* output, int width, int height, int colorWidthStep, int grayWidthStep)
{
// 2D Index of current thread
const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

// Only valid threads perform memory I/O
if ((xIndex < width) && (yIndex < height))
{
//Location of colored pixel in input
const int color_tid = yIndex * colorWidthStep + (3 * xIndex);

//Location of gray pixel in output
const int gray_tid = yIndex * grayWidthStep + xIndex;

const unsigned char blue = input[color_tid];
const unsigned char green = input[color_tid + 1];
const unsigned char red = input[color_tid + 2];

// The standard NTSC conversion formula that is used for calculating the effective luminance of a pixel (https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems)
const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

// Alternatively, use an average
//const float gray = (red + green + blue) / 3.f;

output[gray_tid] = static_cast<unsigned char>(gray);
}
}