#include "includes.h"
__global__ void convolution(uint8_t *inData, uint8_t *outData, int width, int height, float *kernel, int kwidth, int kheight, float ksum)
{
int gx = blockIdx.x * blockDim.x + threadIdx.x;
int gy = blockIdx.y * blockDim.y + threadIdx.y;

if (gx < width && gy < height) {
int rx = (kwidth - 1) / 2;
int ry = (kheight - 1) / 2;

float sum = 0.0;

for (int y = 0; y < kheight; ++y) {
int cy = max(0, min(height - 1, gy + y - ry));

for (int x = 0; x < kwidth; ++x) {
int cx = max(0, min(width - 1, gx + x - rx));
sum = fmaf((float) inData[cx + cy * width], kernel[x + y * kwidth], sum);
}
}

sum = max(0.0, min(255.0, sum));
outData[gx + gy * width] = (uint8_t) fdividef(sum, ksum);
}
}