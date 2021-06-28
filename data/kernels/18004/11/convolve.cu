#include "includes.h"
__global__ void convolve(unsigned char* imgInput, float* mask, int width, int height, int paddingX, int paddingY, int kWidth, int kHeight, unsigned char* imgOutput)
{
// Calculate our pixel's location
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;

float sum = 0.0;
int   pWidth = kWidth / 2;
int   pHeight = kHeight / 2;

//Solo ejecuta validos pixeles
if (x >= pWidth + paddingX && y >= pHeight + paddingY && x < (blockDim.x * gridDim.x) - pWidth - paddingX &&
y < (blockDim.y * gridDim.y) - pHeight - paddingY)
{
for (int j = -pHeight; j <= pHeight; j++)
{
for (int i = -pWidth; i <= pWidth; i++)
{
// Sample the weight for this location
int ki = (i + pWidth);
int kj = (j + pHeight);
float w = mask[(kj * kWidth) + ki];


sum += w * float(imgInput[((y + j) * width) + (x + i)]);
}
}
}
imgOutput[(y * width) + x] = (unsigned char)sum;
}