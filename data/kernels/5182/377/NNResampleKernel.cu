#include "includes.h"
__global__ void NNResampleKernel(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;
int size =  outputWidth * outputHeight;

if (id < size)
{
int px = id % outputWidth;
int py = id / outputWidth;

float xRatio = (float)(inputWidth - 1) / (outputWidth);
float yRatio = (float)(inputHeight - 1) / (outputHeight);

int x = (int) (xRatio * (px+.5f));
int y = (int) (yRatio * (py+.5f));

output[py * outputWidth + px] = input[y*inputWidth + x];
}
}