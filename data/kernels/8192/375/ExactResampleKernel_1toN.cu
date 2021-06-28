#include "includes.h"
__global__ void ExactResampleKernel_1toN(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
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

int xRatio = outputWidth / inputWidth;
int yRatio = outputHeight / inputHeight;

//corresponding coordinates in the original image
int x = px / xRatio;
int y = py / yRatio;

output[py * outputWidth + px] = input[y * inputWidth + x];
}
}