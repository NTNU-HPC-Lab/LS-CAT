#include "includes.h"
__global__ void rgb2gray(float *grayImage, float *rgbImage, int channels, int width, int height) {
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x < width && y < height) {
// get 1D coordinate for the grayscale image
int grayOffset = y * width + x;
// one can think of the RGB image having
// CHANNEL times columns than the gray scale image
int rgbOffset = grayOffset * channels;
float r       = rgbImage[rgbOffset];     // red value for pixel
float g       = rgbImage[rgbOffset + 1]; // green value for pixel
float b       = rgbImage[rgbOffset + 2]; // blue value for pixel
// perform the rescaling and store it
// We multiply by floating point constants
grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
}
}