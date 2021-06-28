#include "includes.h"
__global__ void Crop2DKernel(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int size, int leftMargin, int topMargin, float fillValue)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

if (id < size)
{
int inputX = id % outputWidth - leftMargin;
int inputY = id / outputWidth - topMargin;

if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight)
output[id] = input[inputX + inputY * inputWidth];
else
output[id] = fillValue;
}
}