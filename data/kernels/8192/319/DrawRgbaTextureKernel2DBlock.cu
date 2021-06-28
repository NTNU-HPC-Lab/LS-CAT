#include "includes.h"
__global__ void DrawRgbaTextureKernel2DBlock(float *target, int targetWidth, int targetHeight, int inputX, int inputY, float *texture, int textureWidth, int textureHeight)
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