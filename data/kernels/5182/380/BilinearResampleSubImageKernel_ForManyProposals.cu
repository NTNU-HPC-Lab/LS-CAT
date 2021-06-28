#include "includes.h"
__global__ void BilinearResampleSubImageKernel_ForManyProposals(const float *input, float *output, const float* subImageDefs, bool safeBounds, int subImageDefsDim, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int numberSubImages, int outputSize)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

int px = id % outputWidth;  // line in the single output image
int subim_id = id / outputWidth / outputHeight;  // which image it is
int py = (id / outputWidth) % outputHeight;  // column in the single output image

if (id<outputSize)
{
float subImgCX = subImageDefs[0 + subim_id*subImageDefsDim]; // <-1, 1>
float subImgCY = subImageDefs[1 + subim_id*subImageDefsDim]; // <-1, 1>
float subImgDiameter = subImageDefs[2 + subim_id*subImageDefsDim]; // <0,1>

int maxDiameter = min(inputWidth - 1, inputHeight - 1);
int diameterPix = (int)(subImgDiameter * maxDiameter);

diameterPix = max(1, diameterPix);
diameterPix = min(maxDiameter, diameterPix);

int subImgX = (int)(inputWidth * (subImgCX + 1) * 0.5f) - diameterPix / 2;
int subImgY = (int)(inputHeight * (subImgCY + 1) * 0.5f) - diameterPix / 2;

if (safeBounds)
{
subImgX = max(subImgX, 1);
subImgY = max(subImgY, 1);

subImgX = min(subImgX, inputWidth - diameterPix - 1);
subImgY = min(subImgY, inputHeight - diameterPix - 1);
}

float xRatio = (float)(diameterPix - 1) / (outputWidth - 1);
float yRatio = (float)(diameterPix - 1) / (outputHeight - 1);

int x = (int) (xRatio * px);
int y = (int) (yRatio * py);

if (x + subImgX >= 0 && y + subImgY >= 0 &&
x + subImgX < inputWidth && y + subImgY < inputHeight)
{
//--- X and Y distance difference
float xDist = (xRatio * px) - x;
float yDist = (yRatio * py) - y;

//--- Points
float topLeft= input[(y + subImgY) * inputWidth + x + subImgX];
float topRight = input[(y + subImgY) * inputWidth + x + subImgX + 1];
float bottomLeft = input[(y + subImgY + 1) * inputWidth + x + subImgX];
float bottomRight = input[(y + subImgY + 1) * inputWidth + x + subImgX + 1 ];

float result =
topLeft * (1 - xDist) * (1 - yDist) +
topRight * xDist * (1 - yDist) +
bottomLeft * yDist * (1 - xDist) +
bottomRight * xDist * yDist;

output[py * outputWidth + px + subim_id*outputWidth*outputHeight] = result;
}
}
}