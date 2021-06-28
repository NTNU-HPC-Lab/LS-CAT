#include "includes.h"
__global__ void unMirrorImage_kernel(uint width, uint height, uint border, uint borderWidth, uint borderHeight, float* devInput, float* devOutput) {
int x0 = blockDim.x * blockIdx.x + threadIdx.x;
int y0 = blockDim.y * blockIdx.y + threadIdx.y;
if ((x0 < borderWidth) && (y0 < borderHeight)) {
int x1 = x0 + border;
int y1 = y0 + border;
devOutput[y0 * borderWidth + x0] = devInput[y1 * width + x1];
}
}