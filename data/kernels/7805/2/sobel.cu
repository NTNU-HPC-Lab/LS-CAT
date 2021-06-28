#include "includes.h"




__global__ void sobel(unsigned char *output, unsigned char *input, int width, int height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (y >= height || x >= width)
return;

// Sobel weights
float weightsX[9] = { -1, -2, -1,
0,  0,  0,
1,  2,  1 };

float weightsY[9] = { -1,  0,  1,
-2,  0,  2,
-1,  0,  1 };

int offsetY[9] = { -1,  -1,  -1,
0,   0,   0,
1,   1,   1 };

int offsetX[9] = { -1,   0,   1,
-1,   0,   1,
-1,   0,   1 };


float pointX = 0.f;
float pointY = 0.f;
#pragma unroll
for (int i = 0; i < 9; i++)
{
int index = (x + offsetX[i]) + (y + offsetY[i]) * width;

unsigned char pixel = *(input + index);
pointX += pixel * weightsX[i];
pointY += pixel * weightsY[i];
}


// Do Sobel here!
int index = x + y * width;
unsigned char * outputData = output + index;
outputData[0] = sqrtf(pointX * pointX + pointY * pointY);
}