#include "includes.h"

/// scalar field to upscale
texture<float, cudaTextureType2D, cudaReadModeElementType> texCoarse;
texture<float2, cudaTextureType2D, cudaReadModeElementType> texCoarseFloat2;

__global__
__global__ void TgvUpscaleFloat2Kernel(int width, int height, int stride, float scale, float2 *out)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

if (ix >= width || iy >= height) return;

float x = ((float)ix + 0.5f) / (float)width;
float y = ((float)iy + 0.5f) / (float)height;

// exploit hardware interpolation
// and scale interpolated vector to match next pyramid level resolution
float2 src = tex2D(texCoarseFloat2, x, y);
out[ix + iy * stride].x = src.x * scale;
out[ix + iy * stride].y = src.y * scale;
}