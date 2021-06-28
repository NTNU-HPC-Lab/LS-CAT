#include "includes.h"
__global__ void greyConvertor(unsigned char* output, uchar3 const* input, const uint height, const uint width) {
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;

if (x < width && y < height) {
int grayOffset = y*width + x;
unsigned char r = input[grayOffset].x;
unsigned char g = input[grayOffset].y;
unsigned char b = input[grayOffset].z;
output[grayOffset] = 0.21f*r + 0.72f*g + 0.07f*b;
}
}