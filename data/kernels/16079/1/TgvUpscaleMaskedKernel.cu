#include "includes.h"

/// scalar field to upscale
texture<float, cudaTextureType2D, cudaReadModeElementType> texCoarse;
texture<float2, cudaTextureType2D, cudaReadModeElementType> texCoarseFloat2;

__global__
__global__ void TgvUpscaleMaskedKernel(float * mask, int width, int height, int stride, float scale, float *out)
{
const int ix = threadIdx.x + blockIdx.x * blockDim.x;
const int iy = threadIdx.y + blockIdx.y * blockDim.y;

if ((iy >= height) && (ix >= width)) return;
int pos = ix + iy * stride;
//if (mask[pos] == 0.0f) return;

float x = ((float)ix + 0.5f) / (float)width;
float y = ((float)iy + 0.5f) / (float)height;

// exploit hardware interpolation
// and scale interpolated vector to match next pyramid level resolution
out[pos] = tex2D(texCoarse, x, y) * scale;

//if (ix >= width || iy >= height) return;

//// exploit hardware interpolation
//// and scale interpolated vector to match next pyramid level resolution
//out[ix + iy * stride] = tex2D(texCoarse, x, y) * scale;
}