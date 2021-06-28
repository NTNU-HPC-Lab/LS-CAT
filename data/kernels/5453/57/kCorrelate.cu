#include "includes.h"
__global__ void kCorrelate(float* source, float* kernel, float* dest, int width, int height, int kwidth, int kheight) {
const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numThreads = blockDim.x * gridDim.x;

for (unsigned int i = idx; i < width * height; i += numThreads) {
float sum = 0;
for (int w = -kwidth/2; w <= kwidth/2; w++) {
for (int h = -kheight/2; h <= (kheight)/2; h++) {
const int x = (i / height) + w;
const int y = (i % height) + h;
const int j = i + (w * height) + h;

if (x >= 0 && x < width && y >= 0 && y < height)
sum += source[j] * kernel[(kwidth * kheight / 2) + w * kheight + h];
}
}
dest[i] = sum;
}
}