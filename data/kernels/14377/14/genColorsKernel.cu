#include "includes.h"
__global__ void genColorsKernel(float* colors, int nelems) {
const float AF_BLUE[4]   = {0.0588f, 0.1137f, 0.2745f, 1.0f};
const float AF_ORANGE[4] = {0.8588f, 0.6137f, 0.0745f, 1.0f};

int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < nelems) {
if (i % 2 == 0) {
colors[3 * i + 0] = AF_ORANGE[0];
colors[3 * i + 1] = AF_ORANGE[1];
colors[3 * i + 2] = AF_ORANGE[2];
} else {
colors[3 * i + 0] = AF_BLUE[0];
colors[3 * i + 1] = AF_BLUE[1];
colors[3 * i + 2] = AF_BLUE[2];
}
}
}