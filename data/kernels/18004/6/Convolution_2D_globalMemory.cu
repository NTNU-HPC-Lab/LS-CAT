#include "includes.h"
__global__ void Convolution_2D_globalMemory(unsigned char* imgInput,unsigned char* imgOutput, const float* mask, int height, int width, int channels) {

int Row, Col, filterRow, filterCol;

int rows = threadIdx.x + blockIdx.x * blockDim.x;
int cols = threadIdx.y + blockIdx.y * blockDim.y;
float sum = 0;

Row = rows - MASK_WIDTH /2;
Col = cols - MASK_WIDTH /2;
for (int c = 0; c < channels; c++)
{
sum = 0;
for (int i = 0; i < MASK_WIDTH; i++)
{
for (int j = 0; j < MASK_WIDTH; j++)
{
filterRow = Row + i;
filterCol = Col + j;

if ((filterRow >= 0) && (filterRow < height) && (filterCol >= 0) && (filterCol < width))
{
sum += imgInput[(filterRow * height + filterCol) * channels + c] * mask[i * MASK_WIDTH + j];
}
else { sum = 0; }
}
}
sum/=MASK_WIDTH *MASK_WIDTH ;
imgOutput[(rows * width + cols) * channels + c] = (unsigned char)sum;
}
}