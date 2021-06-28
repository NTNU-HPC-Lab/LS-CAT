#include "includes.h"
__global__ void RgbToRgbaKernel(const uint8_t *__restrict__ input, uint8_t *__restrict__ output, size_t pitch, size_t width_px, size_t height) {
constexpr size_t in_channels = 3, out_channels = 4;
size_t x = threadIdx.x + blockIdx.x * blockDim.x;
size_t y = threadIdx.y + blockIdx.y * blockDim.y;
if (x >= width_px || y >= height) return;
size_t in_idx = in_channels * x + in_channels * width_px * y;
size_t out_idx = out_channels * x + pitch * y;
output[out_idx] = input[in_idx];
output[out_idx + 1] = input[in_idx + 1];
output[out_idx + 2] = input[in_idx + 2];
output[out_idx + 3] = 255;
}