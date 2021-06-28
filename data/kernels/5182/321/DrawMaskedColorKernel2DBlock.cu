#include "includes.h"
__global__ void DrawMaskedColorKernel2DBlock(float *target, int targetWidth, int targetHeight, int inputX, int inputY, float *textureMask, int textureWidth, int textureHeight, float r, float g, float b)
{
int id = blockDim.x * blockDim.y * (blockIdx.y * gridDim.x + blockIdx.x)
+ blockDim.x * threadIdx.y
+ threadIdx.x; // 2D grid of 2D blocks; block dimension x = texture width;
// grid dimension x + block dimension y = texture height

int targetPixels = targetWidth * targetHeight;

int texturePixels = textureWidth * textureHeight;

int idTextureRgb = blockIdx.y;
int idTexturePixel = (id - idTextureRgb * texturePixels);
int idTextureY = blockIdx.x * blockDim.y + threadIdx.y;
int idTextureX = threadIdx.x;


if (idTextureRgb < 3) // only RGB channels are interesting
{
// if the texture pixel offset by inputX, inputY, lies inside the target
if (idTextureX + inputX < targetWidth &&
idTextureX + inputX >= 0 &&
idTextureY + inputY < targetHeight &&
idTextureY + inputY >= 0)
{
int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
int aIndex = idTexturePixel + 3 * texturePixels; // the A component of the texture
float a = textureMask[aIndex];

if (a > 0) // mask allows color here
{
switch (idTextureRgb)
{
case 0:
target[tIndex] = r;
break;
case 1:
target[tIndex] = g;
break;
case 2:
default:
target[tIndex] = b;
break;
}
}
}
}
}