#include "includes.h"
__global__ void initializeToValue_kernel(unsigned int *data, unsigned int value, int width, int height) {
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if (x < width && y < height) {
data[y * width + x] = value;
}
}