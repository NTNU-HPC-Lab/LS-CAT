#include "includes.h"
__global__ void DrawRgbaTextureKernel(float *target, int targetWidth, int targetHeight, int inputX, int inputY, float *texture, int textureWidth, int textureHeight)
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


if (idTextureRgb < 3) // 3 channels that we will write to
{
// the texture is in BGR format, we want RGB
switch (idTextureRgb)
{
case 0: // R
idTextureRgb = 2; // B
break;
case 2: // B
idTextureRgb = 0; // R
break;
}
// if the texture pixel offset by inputX, inputY, lies inside the target
if (idTextureX + inputX < targetWidth &&
idTextureX + inputX >= 0 &&
idTextureY + inputY < targetHeight &&
idTextureY + inputY >= 0)
{
int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
int aIndex = idTexturePixel + 3 * texturePixels; // the A component of the texture
float a = texture[aIndex];
target[tIndex] = target[tIndex] * (1.0f - a) + a * texture[id];
}
}
}