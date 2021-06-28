#include "includes.h"
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char *destination)
{
// Calculate our pixel's location
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;

float sum = 0.0f;
int   pWidth = kWidth/2;
int   pHeight = kHeight/2;

// Execute for valid pixels
if(x >= pWidth+paddingX &&
y >= pHeight+paddingY &&
x < (gridDim.x * blockDim.x)-pWidth-paddingX &&
y < (gridDim.y *blockDim.y )-pHeight-paddingY)
{
for(int j = -pHeight; j <= pHeight; j++)
{
for(int i = -pWidth; i <= pWidth; i++)
{
// Sample the weight for this location
int ki = (i+pWidth);
int kj = (j+pHeight);
float w  = convolutionKernel[(kj * kWidth) + ki + kOffset];


sum += w * float(source[((y+j) * width) + (x+i)]);
}
}
}

// Average the sum
destination[(y * width) + x] = (unsigned char) sum;
}