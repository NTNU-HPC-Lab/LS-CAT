#include "includes.h"
__global__ void ExactResampleKernel_Nto1(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;
int size = outputWidth * outputHeight;

if (id < size)
{
//output point coordinates
int px = id % outputWidth;
int py = id / outputWidth;

int xRatio = inputWidth / outputWidth;
int yRatio = inputHeight / outputHeight;

float sum = 0;
for (int sx = 0; sx < xRatio; sx++) {
for (int sy = 0; sy < yRatio; sy++) {
//corresponding coordinates in the original image
int x = px * xRatio + sx;
int y = py * yRatio + sy;

sum += input[y * inputWidth + x];
}
}

output[py * outputWidth + px] = sum / (float)(xRatio * yRatio);
}
}