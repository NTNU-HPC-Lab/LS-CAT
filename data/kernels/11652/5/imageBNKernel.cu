#include "includes.h"
__global__ void imageBNKernel(unsigned char* d_image, int h, int w)
{
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;
int r, g, b;

if((Row < h) && (Col < w)){
r = d_image[4 * w * Row + 4 * Col + 0];
g = d_image[4 * w * Row + 4 * Col + 1];
b = d_image[4 * w * Row + 4 * Col + 2];

d_image[4 * w * Row + 4 * Col + 0] = 0;
d_image[4 * w * Row + 4 * Col + 1] = 0;
d_image[4 * w * Row + 4 * Col + 2] = 0;
d_image[4 * w * Row + 4 * Col + 3] = (int)(r*0.21 + g*0.71 + b*0.07);
}
}