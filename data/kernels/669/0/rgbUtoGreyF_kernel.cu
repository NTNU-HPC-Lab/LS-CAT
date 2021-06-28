#include "includes.h"
__global__ void rgbUtoGreyF_kernel(int width, int height, unsigned int* rgbU, float* grey) {
int x = blockDim.x * blockIdx.x + threadIdx.x;
int y = blockDim.y * blockIdx.y + threadIdx.y;
if ((x < width) && (y < height)) {
int index = y * width + x;
unsigned int rgb = rgbU[index];
float r = (float)(rgb & 0xff)/255.0;
float g = (float)((rgb & 0xff00) >> 8)/255.0;
float b = (float)((rgb & 0xff0000) >> 16)/255.0;
grey[index] =
(0.29894 * r)
+ (0.58704 * g)
+ (0.11402 * b);
}
}