#include "includes.h"
__global__ void kernGaussianBlur(int width, int height, uint8_t * dst, uint8_t * src) {
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;
if (x >= width || y >= height) {
return;
}
float kernel[5][5] = {
{0.003765, 0.015019, 0.023792, 0.015019, 0.003765},
{0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
{0.023792, 0.094907, 0.150342, 0.094907, 0.023792},
{0.015019, 0.059912, 0.094907, 0.059912, 0.015019},
{0.003765, 0.015019, 0.023792, 0.015019, 0.003765}
};
float r, g, b;
r = g = b = 0.0;
for (int i = 0; i < 5; i++) {
int tx = x + i - 2;
for (int j = 0; j < 5; j++) {
int ty = y + j - 2;
if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
r += src[(ty * width + tx) * 3] * kernel[i][j];
g += src[(ty * width + tx) * 3 + 1] * kernel[i][j];
b += src[(ty * width + tx) * 3 + 2] * kernel[i][j];
}
}
}
int idx = 3 * (y * width + x);
dst[idx] = r;
dst[idx + 1] = g;
dst[idx + 2] = b;
return;
}