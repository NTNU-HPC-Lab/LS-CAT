#include "includes.h"
__global__ void rgb2gray (float * input, float *output, int height, int width)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if(x<height && y<width)
{
unsigned int idx = x* width + y;
float r          = input[3 * idx];
float g          = input[3 * idx + 1]; // green value for pixel
float b          = input[3 * idx + 2];
output[idx] = (0.21f * r + 0.71f * g + 0.07f * b);
}
}