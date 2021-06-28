#include "includes.h"
__global__ void shiftLeftPixels(int16_t *bayImg, size_t width, size_t height, int bppMult)
{
int2 pixelCoord = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

if (pixelCoord.x < width && pixelCoord.y < height)
{
bayImg[pixelCoord.y * width + pixelCoord.x] <<= bppMult;
}
}