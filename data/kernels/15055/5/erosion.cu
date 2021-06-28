#include "includes.h"
__global__ void erosion(uint8_t *inData, uint8_t *outData, int radiusX, int radiusY, int width, int height)
{
int gx = blockIdx.x * blockDim.x + threadIdx.x;
int gy = blockIdx.y * blockDim.y + threadIdx.y;

int x1 = gx - radiusX;
int x2 = gx + radiusX;
int y1 = gy - radiusY;
int y2 = gy + radiusY;

if (x1 < 0) {
x1 = 0;
} else if (x2 >= width) {
x2 = width - 1;
}

if (y1 < 0) {
y1 = 0;
} else if (y2 >= height) {
y2 = height - 1;
}

uint8_t minimum = 255;

for (int y = y1; y <= y2; ++y) {
for (int x = x1; x <= x2; ++x) {
minimum = min(minimum, inData[width * y + x]);
}
}

outData[width * gy + gx] = minimum;
}