#include "includes.h"
__global__ void fillImage(int width, int height, int value, int* devOutput) {
int x = blockDim.x * blockIdx.x + threadIdx.x;
int y = blockDim.y * blockIdx.y + threadIdx.y;
int index = y * width + x;
if ((y < height) && (x < width)) {
devOutput[index] = value;
}
}