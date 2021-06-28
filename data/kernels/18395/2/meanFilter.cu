#include "includes.h"
__global__ void meanFilter(unsigned char *input, unsigned char *output, int height, int width)
{
int col= blockDim.x * blockIdx.x + threadIdx.x;
int row = blockDim.y * blockIdx.y + threadIdx.y;
if (row < height && col < width)
{
int pixelIndex = row * width + col;
int pixelNum = 0;
int tempSum = 0;
for(int i = -FILTER_SIZE + 1; i <  FILTER_SIZE; i++)
{
for(int j = -FILTER_SIZE + 1; j < FILTER_SIZE; j++ )
{
if(col + i >= 0 && col + i < width && row + j >= 0 && row + j < height)
{
tempSum += input[(row + j) * width + col +i];
pixelNum++;
}
}
}
output[pixelIndex] = tempSum/pixelNum;
}
}