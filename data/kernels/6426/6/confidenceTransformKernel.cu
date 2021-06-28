#include "includes.h"
__global__ void confidenceTransformKernel(const int width, const int height, const float threshold, const float gamma, const float clampedValue, const float* inputConfidence, float* outputConfidence) {
uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
if (x >= width || y >= height) return;
float inputValue = inputConfidence[y * width + x];
if (inputValue < threshold) {
outputConfidence[y * width + x] = 0;
} else {
outputConfidence[y * width + x] = powf(inputValue, gamma);
}
}