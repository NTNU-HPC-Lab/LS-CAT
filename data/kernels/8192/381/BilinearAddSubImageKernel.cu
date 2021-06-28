#include "includes.h"
__global__ void BilinearAddSubImageKernel(float *input, float *opImage, float* subImageDefs, int inputWidth, int inputHeight, int opImageWidth, int opImageHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

float subImgCX = subImageDefs[0]; // <-1, 1>
float subImgCY = subImageDefs[1]; // <-1, 1>
float subImgDiameter = subImageDefs[2]; // <0,1>

int maxDiameter = min(inputWidth, inputHeight);
int diameterPix = (int)(subImgDiameter * maxDiameter);
diameterPix = max(1, diameterPix);

int subImgX = (int)(inputWidth * (subImgCX + 1) * 0.5f) - diameterPix / 2;
int subImgY = (int)(inputHeight * (subImgCY + 1) * 0.5f) - diameterPix / 2;

int px = id % diameterPix;
int py = id / diameterPix;

if (px + subImgX >= 0 && py + subImgY >= 0 &&
px + subImgX < inputWidth && py + subImgY < inputHeight &&
py < diameterPix )
{
float xRatio = (float)(opImageWidth - 1) / (diameterPix);
float yRatio = (float)(opImageHeight - 1) / (diameterPix);

int x = (int) (xRatio * px);
int y = (int) (yRatio * py);

// X and Y distance difference
float xDist = (xRatio * px) - x;
float yDist = (yRatio * py) - y;

// Points
float topLeft= opImage[y * opImageWidth + x];
float topRight = opImage[y * opImageWidth + x + 1];
float bottomLeft = opImage[(y + 1) * opImageWidth + x];
float bottomRight = opImage[(y + 1) * opImageWidth + x + 1];

float result =
topLeft * (1 - xDist) * (1 - yDist) +
topRight * xDist * (1 - yDist) +
bottomLeft * yDist * (1 - xDist) +
bottomRight * xDist * yDist;


input[(py + subImgY) * inputWidth + px + subImgX] += result;
}
}