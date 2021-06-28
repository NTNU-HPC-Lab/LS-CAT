#include "includes.h"
__global__ void DrawRgbBackgroundKernel(float *target, int inputWidth, int inputHeight, float r, float g, float b)
{
int column = threadIdx.x + blockDim.x * blockIdx.z;
if (column >= inputWidth)
return;

int id = inputWidth * ( blockIdx.y * gridDim.x + blockIdx.x) // blockIdx.x == row, blockIdx.y == color channel
+ column;

int imagePixels = inputWidth * inputHeight;

if (id < 3*imagePixels) // 3 for RGB
{
float color = 0.0f;
switch (blockIdx.y)
{
case 0:
color = r;
break;
case 1:
color = g;
break;
case 2:
color = b;
break;
}
target[id] = color;
}
}