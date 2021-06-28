#include "includes.h"
__global__ void quantizeImage_kernel(uint width, uint height, uint nbins, float* devInput, int* devOutput) {
int x0 = blockDim.x * blockIdx.x + threadIdx.x;
int y0 = blockDim.y * blockIdx.y + threadIdx.y;
if ((x0 < width) && (y0 < height)) {
int index = y0 * width + x0;
float input = devInput[index];
int output = (int)floorf(input * (float)nbins);
if (output == nbins) {
output = nbins - 1;
}
devOutput[index] = output;
}
}