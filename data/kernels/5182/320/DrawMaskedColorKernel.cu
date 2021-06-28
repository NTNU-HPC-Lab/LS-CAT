#include "includes.h"
__global__ void DrawMaskedColorKernel(float *target, int targetWidth, int targetHeight, int inputX, int inputY, float *textureMask, int textureWidth, int textureHeight, float r, float g, float b)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

int targetPixels = targetWidth * targetHeight;

int texturePixels = textureWidth * textureHeight;

int idTextureRgb = id / texturePixels;
int idTexturePixel = (id - idTextureRgb * texturePixels); // same as (id % texturePixels), but the kernel runs 10% faster
int idTextureY = idTexturePixel / textureWidth;
int idTextureX = (idTexturePixel - idTextureY * textureWidth); // same as (id % textureWidth), but the kernel runs another 10% faster

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