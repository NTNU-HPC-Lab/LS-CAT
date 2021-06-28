#include "includes.h"
__global__ void kernGaussianBlur(int width, int height, uint8_t * dst, uint8_t * src, int kernSize, float * kernel) {
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;
if (x >= width || y >= height) {
return;
}

float r, g, b;
r = g = b = 0.0;
for (int i = 0; i < kernSize; i++) {
int tx = x + i - kernSize/2;
for (int j = 0; j < kernSize; j++) {
int ty = y + j - kernSize/2;
if (tx >= 0 && ty >= 0 && tx < width && ty < height) {
r += src[(ty * width + tx) * 3] * kernel[j * kernSize + i];
g += src[(ty * width + tx) * 3 + 1] * kernel[j * kernSize + i];
b += src[(ty * width + tx) * 3 + 2] * kernel[j * kernSize + i];
}
}
}
int idx = 3 * (y * width + x);
dst[idx] = r;
dst[idx + 1] = g;
dst[idx + 2] = b;
return;
}