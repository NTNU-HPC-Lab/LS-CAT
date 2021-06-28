#include "includes.h"
__global__ void erosion2(uint8_t *inData, uint8_t *outData, int radiusX, int radiusY, int width, int height)
{
__shared__ uint8_t localData[TILE_SIZE * TILE_SIZE];

int tx = threadIdx.x;
int ty = threadIdx.y;
int gx = blockIdx.x * blockDim.x;
int gy = blockIdx.y * blockDim.y;

localData[TILE_SIZE * (radiusY + ty) + radiusX + tx] = inData[width * (gy + ty) + gx + tx];

int x1 = tx, y1 = ty, x2 = tx, y2 = ty;

if (tx == 0) {
x1 = max(0, gx - radiusX) - gx;
} else if (tx == blockDim.x - 1) {
x2 = min(width - 1 - gx, blockDim.x + radiusX - 1);
}

if (ty == 0) {
y1 = max(0, gy - radiusY) - gy;
} else if (ty == blockDim.y - 1) {
y2 = min(height - 1 - gy, blockDim.y + radiusY - 1);
}

__syncthreads();

for (int y = y1; y <= y2; ++y) {
for (int x = x1; x <= x2; ++x) {
localData[TILE_SIZE * (radiusY + y) + radiusX + x] = inData[width * (gy + y) + gx + x];
}
}

__syncthreads();

x1 = tx - radiusX;
x2 = tx + radiusX;
y1 = ty - radiusY;
y2 = ty + radiusY;

if (gx + x1 < 0) {
x1 = 0;
} else if (gx + x2 >= width) {
x2 = width - gx - 1;
}

if (gy + y1 < 0) {
y1 = 0;
} else if (gy + y2 >= height) {
y2 = height - gy - 1;
}

uint8_t minimum = 255;

for (int y = y1; y <= y2; ++y) {
for (int x = x1; x <= x2; ++x) {
minimum = min(minimum, localData[TILE_SIZE * (radiusY + y) + radiusX + x]);
}
}

outData[width * (gy + ty) + gx + tx] = minimum;
}