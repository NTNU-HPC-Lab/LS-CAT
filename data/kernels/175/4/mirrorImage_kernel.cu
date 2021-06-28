#include "includes.h"
// vim: ts=4 syntax=cpp comments=


#define XBLOCK 16
#define YBLOCK 16





__global__ void mirrorImage_kernel(uint width, uint height, uint border, uint borderWidth, uint borderHeight, int* devInput, int* devOutput) {
int x0 = blockDim.x * blockIdx.x + threadIdx.x;
int y0 = blockDim.y * blockIdx.y + threadIdx.y;
if ((x0 < borderWidth) && (y0 < borderHeight)) {
int x1 = 0;
int y1 = 0;
if (x0 < border) {
x1 = border - x0 - 1;
} else if (x0 < border + width) {
x1 = x0 - border;
} else {
x1 = border + 2 * width - x0 - 1;
}
if (y0 < border) {
y1 = border - y0 - 1;
} else if (y0 < border + height) {
y1 = y0 - border;
} else {
y1 = border + 2 * height - y0 - 1;
}
devOutput[y0 * borderWidth + x0] = devInput[y1 * width + x1];
}
}