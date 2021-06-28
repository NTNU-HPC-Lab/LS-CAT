#include "includes.h"
__global__ void DrawRgbaColorKernel(float *target, int targetWidth, int targetHeight, int inputX, int inputY, int areaWidth, int areaHeight, float r, float g, float b)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

int targetPixels = targetWidth * targetHeight;

int texturePixels = areaWidth * areaHeight;

int idTextureRgb = id / texturePixels;
int idTexturePixel = (id - idTextureRgb * texturePixels); // same as (id % texturePixels), but the kernel runs 10% faster
int idTextureY = idTexturePixel / areaWidth;
int idTextureX = (idTexturePixel - idTextureY * areaWidth); // same as (id % textureWidth), but the kernel runs another 10% faster


if (idTextureRgb < 3) // 3 channels that we will write to
{
// if the texture pixel offset by inputX, inputY, lies inside the target
if (idTextureX + inputX < targetWidth &&
idTextureX + inputX >= 0 &&
idTextureY + inputY < targetHeight &&
idTextureY + inputY >= 0)
{
float color = 0.0f;
switch (idTextureRgb)
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
int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
target[tIndex] = color;
}
}
}