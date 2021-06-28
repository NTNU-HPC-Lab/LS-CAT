#include "includes.h"
__global__ void ConvertGrayToYCbCr8uKernel(const uint8_t *input, uint8_t *output, unsigned int total_pixels) {
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= total_pixels) {
return;
}

const uint8_t pixel_in = input[idx];
const unsigned int C = 3;
uint8_t* pixel_out = &output[idx * C];
pixel_out[0] = pixel_in;
pixel_out[1] = 128;
pixel_out[2] = 128;
}