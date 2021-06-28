#include "includes.h"
__global__ void DrawRgbaTextureKernelNearestNeighbor(float *target, int targetWidth, int targetHeight, int inputX, int inputY, float *texture, int textureWidth, int textureHeight, int objectWidth, int objectHeight)
{
int id = blockDim.x * blockIdx.y * gridDim.x
+ blockDim.x * blockIdx.x
+ threadIdx.x;

int targetPixels = targetWidth * targetHeight;

int texturePixels = textureWidth * textureHeight;

int objectPixels = objectWidth * objectHeight;

int idObjectRgb = id / objectPixels;
int idObjectPixel = (id - idObjectRgb * objectPixels); // same as (id % objectPixels), but the kernel runs 10% faster
int idObjectY = idObjectPixel / objectWidth;
int idObjectX = (idObjectPixel - idObjectY * objectWidth); // same as (id % textureWidth), but the kernel runs another 10% faster


if (idObjectRgb < 3) // 3 channels that we will write to
{
int targetRgb = idObjectRgb;
// the texture is in BGR format, we want RGB
switch (idObjectRgb)
{
case 0: // R
targetRgb = 2; // B
break;
case 2: // B
targetRgb = 0; // R
break;
}
// if the object pixel offset by inputX, inputY, lies inside the target
if (idObjectX + inputX < targetWidth &&
idObjectX + inputX >= 0 &&
idObjectY + inputY < targetHeight &&
idObjectY + inputY >= 0)
{
// nearest neighbor texture X,Y:
int textureX = textureWidth * idObjectX / objectWidth;
int textureY = textureHeight * idObjectY / objectHeight;
int textureId = textureY * textureWidth + textureX;

int rgbIndex = textureId + idObjectRgb * texturePixels;
float textureValue = texture[rgbIndex];

int tIndex = targetPixels * targetRgb + targetWidth * (idObjectY + inputY) + (idObjectX + inputX);
int aIndex = textureId + 3 * texturePixels; // the A component of the texture
float a = texture[aIndex];
target[tIndex] = target[tIndex] * (1.0f - a) + a * textureValue;
}
}
}