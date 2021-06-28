#include "includes.h"




__global__ void toGrayScale(unsigned char *output, unsigned char *input, int width, int height, int components)
{
int column = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

if (row >= height || column >= width)
return;

int index = column + row * width;
unsigned char * threadData = input + components * index;
unsigned char * outputData = output + index;

const float partRed = 0.299f;
const float partGreen = 0.587f;
const float partBlue = 0.114;

unsigned char greyScale = partBlue * threadData[0] + partGreen * threadData[1] + partRed * threadData[2];

outputData[0] = greyScale;
}