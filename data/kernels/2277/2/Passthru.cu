#include "includes.h"
__device__ uint32_t RGBAPACK_8bit(float red, float green, float blue, uint32_t alpha)
{
uint32_t ARGBpixel = 0;

// Clamp final 10 bit results
red   = min(max(red,   0.0f), 255.0f);
green = min(max(green, 0.0f), 255.0f);
blue  = min(max(blue,  0.0f), 255.0f);

// Convert to 8 bit unsigned integers per color component
ARGBpixel = ((((uint32_t)red)   << 24) |
(((uint32_t)green) << 16) |
(((uint32_t)blue)  <<  8) | (uint32_t)alpha);

return  ARGBpixel;
}
__global__ void Passthru(uint32_t *srcImage,   size_t nSourcePitch, uint32_t *dstImage,   size_t nDestPitch, uint32_t width,       uint32_t height)
{
int x, y;
uint32_t yuv101010Pel[2];
uint32_t processingPitch = ((width) + 63) & ~63;
uint32_t dstImagePitch   = nDestPitch >> 2;
uint8_t *srcImageU8     = (uint8_t *)srcImage;

processingPitch = nSourcePitch;

// Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
y = blockIdx.y *  blockDim.y       +  threadIdx.y;

if (x >= width)
return; //x = width - 1;

if (y >= height)
return; // y = height - 1;

// Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
// if we move to texture we could read 4 luminance values
yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]);
yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]);

// this steps performs the color conversion
float luma[2];

luma[0]   = (yuv101010Pel[0]        & 0x00FF);
luma[1]   = (yuv101010Pel[1]        & 0x00FF);

// Clamp the results to RGBA
dstImage[y * dstImagePitch + x     ] = RGBAPACK_8bit(luma[0], luma[0], luma[0], constAlpha);
dstImage[y * dstImagePitch + x + 1 ] = RGBAPACK_8bit(luma[1], luma[1], luma[1], constAlpha);
}